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

import sys
import time
import traceback
from edl.utils import args_utils
from edl.utils import constants
from edl.utils import env as edl_env
from edl.utils import etcd_db
from edl.utils import exceptions
from edl.utils import log_utils
from edl.utils import pod_server_client
from edl.utils import status as edl_status
from edl.utils import train_process as edl_train_process
from edl.utils import resource_pods

from edl.utils import leader_pod
from ..utils.log_utils import logger
from ..utils.pod import Pod
from ..utils.pod_server import PodServer
from ..utils.register import PodResourceRegister
from ..utils.watcher import Watcher


def edl_barrier(job_env, pod, timeout):
    start = time.time()

    log_time = time.time()
    while True:
        try:
            etcd = etcd_db.get_global_etcd()
            leader = leader_pod.load_from_etcd(etcd)
            if leader is None:
                raise exceptions.EdlNotFoundLeader("can't get leader")

            logger.debug("barrier on leader:{}".format(leader))

            c = pod_server_client.PodServerClient(leader.endpoint)
            cluster = c.barrier(job_env.job_id, pod.get_id())
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


def prepare(args):
    args_dict = args_utils.convert_args_to_dict(args)

    # job enviroment.
    job_env = edl_env.JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))

    # get global etcd and lock
    etcd = etcd_db.get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    last_status = edl_status.load_job_status_from_etcd(etcd)
    if last_status == edl_status.Status.SUCCEED:
        logger.info("job:{} has completed! Need't try!".format(job_env.job_id))
        sys.exit(0)

    # local pod, and the pod's id does't change.
    pod = Pod()
    pod.from_env(job_env)

    # update pod status
    edl_status.save_pod_status_to_etcd(etcd,
                                       pod.get_id(), edl_status.Status.INITIAL)

    # launch pod server
    pod_server = PodServer(job_env, pod.get_id())
    pod_server.start(job_env, pod)
    logger.info("pod server started:[{}]".format(pod))

    return job_env, pod, pod_server


def job_exit(cluster,
             leader_register,
             resource_register,
             watcher,
             pod,
             trainer_flag,
             register_flag,
             barrier_flag,
             resource_flag,
             timeout=300):
    local_flag = trainer_flag & register_flag & barrier_flag
    etcd = etcd_db.get_global_etcd()
    edl_status.save_pod_flag_to_ecd(etcd, pod.get_id(), local_flag)

    begin = time.time()
    while True:
        try:
            if leader_register.is_leader():
                if etcd.wait_resource(cluster, timeout=15):
                    job_flag = trainer_flag & register_flag & barrier_flag & resource_flag
                    edl_status.save_job_flag_to_etcd(etcd, job_flag)
                    logger.info("set job status:{} ok!".format(job_flag))
                    break
                raise exceptions.EdlWaitFollowersReleaseError(
                    "can't wait resource")
            else:
                break
        except Exception as e:
            logger.warning("prepare job_exit meets error:{}".format(e))
            if time.time() - begin >= timeout:
                logger.warning("wait resource error")
                break

            time.sleep(3)
            continue

    leader_register.stop()
    watcher.stop()
    resource_register.stop()


def launch(args):
    job_env, pod, pod_server = prepare(args)

    # register pod resource to tell others:
    # this resource can use to train
    resource_register = resource_pods.Register(job_env, pod)

    # seize the leader
    leader_register = leader_pod.Register(job_env, pod.get_id())

    # register rank and watch the rank
    # if the rank changed, the pods should restart the training proc.
    # pod exit if barrier error
    cluster = edl_barrier(job_env, pod, timeout=600)

    # update pod status
    etcd = etcd_db.get_global_etcd()
    edl_status.save_pod_status_to_etcd(etcd,
                                       pod.get_id(), edl_status.Status.RUNNING)

    # watcher after barrier
    watcher = Watcher(job_env, cluster, pod)

    procs = edl_train_process.start(
        cluster,
        pod,
        args.training_script,
        args.training_script_args,
        log_dir=args.log_dir)

    trainer_flag = True
    register_flag = True
    barrier_flag = True
    while True:
        # check local status first
        alive, trainer_flag = edl_train_process.watch(procs, pod.trainers_num)
        if not alive or not trainer_flag:
            break

        if resource_register.is_stopped() or leader_register.is_stopped():
            edl_train_process.terminate()
            register_flag = False
            break

        # check job status second
        if watcher.changed:
            new_cluster = edl_barrier(job_env, pod, timeout=60)
            if not new_cluster:
                barrier_flag = False
                break

            edl_train_process.terminate(procs)

            cluster = new_cluster
            watcher = Watcher(job_env, cluster, pod)

            procs = edl_train_process.start(
                job_env,
                cluster,
                pod,
                args.training_script,
                args.training_script_args,
                log_dir=args.log_dir)

        time.sleep(3)

    if not register_flag:
        logger.fatal("register meets error and local exit!")

    if not leader_register.is_leader():
        leader_register.stop()

    job_exit(
        cluster=cluster,
        leader_register=leader_register,
        resource_register=resource_register,
        watcher=watcher,
        pod=pod,
        trainer_flag=trainer_flag,
        register_flag=register_flag,
        barrier_flag=barrier_flag)


def main():
    log_utils.get_logger(log_level=10)
    args = args_utils.parse_args()
    launch(args)


if __name__ == '__main__':
    main()
