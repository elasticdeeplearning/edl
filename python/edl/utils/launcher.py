# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from edl.utils import cluster_generator


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
        self._procs = None

        self._trainer_flag = True
        self._resource_register_flag = True
        self._leader_register_flag = True
        self._barrier_flag = True
        self._args = None

    def init(self):
        # update pod status
        edl_status.save_pod_status_to_etcd(
            self._etcd, self._pod.get_id(), edl_status.Status.INITIAL, timeout=30
        )

        # launch pod server
        self._pod_server = pod_server.PodServer(self._job_env, self._pod)
        self._pod_server.start()
        logger.info("pod server started:[{}]".format(self._pod))

    def _barrier(self, timeout):
        log_time = time.time()
        start = log_time
        leader = None
        while True:
            try:
                leader = leader_pod.load_from_etcd(self._etcd, timeout=15)
                if leader is None:
                    raise exceptions.EdlNotFoundLeader("can't get leader")

                logger.debug("barrier on leader:{}".format(leader))

                client = pod_server_client.Client(leader.endpoint)
                cluster = client.barrier(self._job_env.job_id, self._pod.get_id())
                return cluster
            except Exception as e:
                if time.time() - log_time > 30:
                    logger.info("wait to barrier now!")
                    log_time = time.time()
                logger.debug("barrier error:{} {}".format(e, traceback.format_exc()))

            if time.time() - start > timeout:
                message = "wait to barrier with all error:{} \
                    leader:[{}] current pod:[{}]".format(
                    traceback.format_exc(), leader, self._pod.pod_id
                )
                raise exceptions.EdlBarrierError(message)

            time.sleep(3)

    @error_utils.handle_errors_until_timeout
    def _exit(self, timeout=60):
        if not self._leader_register_flag:
            logger.fatal("leader_register meets error and local pod exit!")

        if not self._resource_register:
            logger.fatal("resource_register meets error and local pod exit!")

        if not self._trainer_flag:
            logger.fatal("local_trainers meets error and local pod exit!")

        local_flag = (
            self._trainer_flag
            & self._leader_register_flag
            & self._barrier_flag
            & self._resource_register_flag
        )
        edl_status.save_pod_flag_to_etcd(
            etcd=self._etcd, pod_id=self._pod.get_id(), flag=local_flag, timeout=15
        )

        if self._leader_register is not None and self._leader_register.is_leader():
            if resource_pods.wait_resource(
                etcd=self._etcd, pod_id=self._pod.pod_id, timeout=60
            ):
                job_flag = local_flag & self._barrier_flag
                edl_status.save_job_flag_to_etcd(
                    etcd=self._etcd, pod_id=self._pod.pod_id, flag=job_flag, timeout=15
                )
                logger.info("set job status:{} ok!".format(job_flag))

        logger.info("end _exit")

    def launch(self):
        """
        let this program can exit normallly
        """
        try:
            self._launch()
            self._exit(timeout=30)
        finally:
            self.__exit__()

    def _check_and_update_local_pod(self):
        pods_ids = self._cluster.get_pods_ids_set()
        if self._pod.pod_id not in pods_ids:
            logger.info(
                "self pod_id:{} not in cluster:{}, so this pod exit!".format(
                    self._pod.pod_id, pods_ids
                )
            )
            return False
        self._pod = self._cluster.get_pod_by_id(self._pod.pod_id)
        logger.info("update local pod:{}".format(self._pod))

        return True

    def _terminate_local_procs(self):
        edl_train_process.terminate(self._procs)
        self._procs = None

    def _launch(self):
        self._resource_register = resource_pods.Register(
            job_env=self._job_env, pod_id=self._pod.pod_id, pod_json=self._pod.to_json()
        )

        generator = cluster_generator.Generator(
            job_env=self._job_env, pod_id=self._pod.get_id()
        )

        self._leader_register = leader_pod.Register(
            job_env=self._job_env,
            pod_id=self._pod.get_id(),
            cluster_generator=generator,
        )

        self._cluster = self._barrier(timeout=600)

        if not self._check_and_update_local_pod():
            return

        # update pod status
        edl_status.save_pod_status_to_etcd(
            self._etcd, self._pod.get_id(), edl_status.Status.RUNNING, timeout=15
        )

        # watcher after barrier
        self._watcher = cluster_watcher.Watcher(
            job_env=self._job_env, cluster=self._cluster
        )

        self._procs = edl_train_process.start(
            job_env=self._job_env,
            cluster=self._cluster,
            pod=self._pod,
            training_script=self._args.training_script,
            training_script_args=self._args.training_script_args,
            log_dir=self._args.log_dir,
        )

        self._trainer_flag = True
        self._register_flag = True
        self._barrier_flag = True
        while True:
            # check local status first
            alive, self._trainer_flag = edl_train_process.watch(
                self._procs, self._pod.trainers_num
            )
            if not alive or not self._trainer_flag:
                break

            if self._resource_register.is_stopped():
                self._terminate_local_procs()
                self._resource_register_flag = False
                break

            if self._leader_register.is_stopped():
                self._terminate_local_procs()
                self._leader_register_flag = False
                break

            # check job status second
            if self._watcher.changed:
                new_cluster = self._barrier(timeout=60)
                if not new_cluster:
                    self._barrier_flag = False
                    break

                self._terminate_local_procs()

                self._cluster = new_cluster
                if not self._check_and_update_local_pod():
                    return

                self._watcher = cluster_watcher.Watcher(
                    job_env=self._job_env, cluster=self._cluster
                )

                self._procs = edl_train_process.start(
                    job_env=self._job_env,
                    cluster=self._cluster,
                    pod=self._pod,
                    training_script=self._args.training_script,
                    training_script_args=self._args.training_script_args,
                    log_dir=self._args.log_dir,
                )

            time.sleep(3)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._leader_register is not None:
            self._leader_register.stop()

        if self._resource_register is not None:
            self._resource_register.stop()

        if self._watcher is not None:
            self._watcher.stop()

        if self._procs:
            edl_train_process.terminate(self._procs)

        self.__initial__()
