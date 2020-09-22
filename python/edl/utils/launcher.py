# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
paddle.distributed.launch is a module that spawns multiple distributed
process on each training node for gpu training.
"""

from __future__ import print_function

import time
import traceback
from edl.utils import exceptions
from edl.utils import pod_server_client
from edl.utils import status as edl_status
from edl.utils import train_process as edl_train_process

from edl.utils import leader_pod
from edl.utils.log_utils import logger
from edl.utils import pod_server
from edl.utils import resource_pods
from edl.utils import cluster_watcher
from edl.utils import error_utils


class Launcher(object):
    def __init__(self, job_env, pod, etcd, args):
        self.__initial__()
        self._job_env = job_env
        self._pod = pod
        self._args = args
        self._etcd = etcd

    def __initial__(self):
        self._job_env = None
        self._pod = None
        self._pod_server = None
        self._cluster = None
        self._leader_register = None
        self._resource_register = None
        self._watcher = None
        self._etcd = None
        self._procs=None

        self._trainer_flag  = True
        self._register_flag = True
        self._barrier_flag  = True
        self._args = None


    def init(self):
        # update pod status
        edl_status.save_pod_status_to_etcd(self._etcd,
                                           self._pod.get_id(), edl_status.Status.INITIAL)

        # launch pod server
        self._pod_server = pod_server.PodServer(self._job_env, self._pod.get_id())
        self._pod_server.start(self._job_env, self._pod)
        logger.info("pod server started:[{}]".format(self._pod))

    def _barrier(self, timeout):
        log_time = time.time()
        start = log_time
        while True:
            try:
                leader = leader_pod.get_pod_leader(self._etcd)
                if leader is None:
                    raise exceptions.EdlNotFoundLeader("can't get leader")

                logger.debug("barrier on leader:{}".format(leader))

                c = pod_server_client.PodServerClient(leader.endpoint)
                cluster = c.barrier(self._job_env.job_id, self._pod.get_id())
                return cluster
            except Exception as e:
                if time.time() - log_time > 30:
                    logger.info("wait to barrier now!")
                    log_time = time.time()
                logger.debug("barrier error:{} {}".format(e,
                                                          traceback.format_exc()))

            if time.time() - start > timeout:
                message = "wait to barrier with all error:{} leader:[{}] current pod:[{}]".format(
                    traceback.format_exc(), leader, pod)
                raise exceptions.EdlBarrierError(message)

            time.sleep(3)

    @error_utils.handle_errors_until_timeout
    def _exit(self, timeout=60):
        local_flag = self._trainer_flag & self._register_flag & self._barrier_flag
        edl_status.save_pod_flag_to_ecd(self._etcd, self._pod.get_id(), local_flag)

        if self._leader_register is not None and self._leader_register.is_leader():
            if resource_pods.wait_resource(self._cluster, timeout=15):
                job_flag = self._trainer_flag & self._register_flag & self._barrier_flag & self._resource_flag
                edl_status.save_job_flag_to_etcd(self._etcd, job_flag)
                logger.info("set job status:{} ok!".format(job_flag))

        if not self._leader_register_flag:
            logger.fatal("leader_register meets error and local pod exit!")

        if not self._resource_register:
            logger.fatal("resource_register meets error and local pod exit!")

        if not self._trainer_flag:
            logger.fatal("local_trainers meets error and local pod exit!")

    def launch(self):
        """
        let this program can exit normallly
        """
        try:
            self._launch()
            self._exit(timeout=30)
        except Exceptions as e:
            raise e
        finally:
            self.__exit__()

    def _launch(self):
        self._resource_register = resource_pods.Register(self._job_env, self._pod)
        self._leader_register = leader_pod.Register(self._job_env, self._pod.get_id())
        self._cluster = self._edl_barrier(self._job_env, self._pod, timeout=600)

        # update pod status
        edl_status.save_pod_status_to_etcd(self._etcd,
                                           self._pod.get_id(), edl_status.Status.RUNNING)

        # watcher after barrier
        self._watcher = cluster_watcher.Watcher(self._job_env, self._cluster, self._pod)

        self._procs = edl_train_process.start(
            self._cluster,
            self._pod,
            self._args.training_script,
            self._args.training_script_args,
            log_dir=self._args.log_dir)

        self._trainer_flag = True
        self._register_flag = True
        self._barrier_flag = True
        while True:
            # check local status first
            alive, self._trainer_flag = edl_train_process.watch(self._procs, self._pod.trainers_num)
            if not alive or not self._trainer_flag:
                break

            if self._resource_register.is_stopped():
                edl_train_process.terminate(self._procs)
                self._resource_register_flag = False
                break

            if self._leader_register.is_stopped():
                edl_train_process.terminate(self._procs)
                self._leader_register_flag = False
                break

            # check job status second
            if self._watcher.changed:
                new_cluster = self._edl_barrier(self._job_env, self._pod, timeout=60)
                if not new_cluster:
                    self._barrier_flag = False
                    break

                edl_train_process.terminate(self._procs)

                self._cluster = new_cluster
                self._watcher = cluster_watcher.Watcher(self._job_env, self._cluster, self._pod)

                self._procs = edl_train_process.start(
                    self._job_env,
                    self._cluster,
                    self._pod,
                    self._args.training_script,
                    self._args.training_script_args,
                    log_dir=self._args.log_dir)

            time.sleep(3)



    def __exit__(self):
        if self._leader_register is not None:
            self._leader_register.stop()

        if self._resource_register is not None:
            self._resource_register.stop()

        if self._watcher is not None:
            self._watcher.stop()

        if self._procs:
            edl_train_process.terminate(self._procs)

        self.__initial__()

