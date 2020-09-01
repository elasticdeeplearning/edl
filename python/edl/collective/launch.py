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
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle.fluid as fluid
from contextlib import closing
import socket
import traceback

from ..utils.edl_env import JobEnv
from ..utils.pod import Pod
from ..utils.register import PodResourceRegister
from ..utils.leader_register import LeaderRegister
from ..utils.etcd_db import get_global_etcd
from ..utils.watcher import Watcher
from ..utils.pod_server import PodServer
from ..utils.utils import logger
from ..utils import utils
from ..utils.global_vars import *
from ..utils.pod_client import PodServerClient
from ..utils.exceptions import *
from ..utils.edl_process import start_local_trainers, terminate_local_procs, watch_local_trainers


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.''')

    parser.add_argument("--nodes_range", type=str, default=None, help="")

    parser.add_argument("--nproc_per_node", type=int, default=None, help="")

    parser.add_argument(
        "--etcd_endpoints", type=str, default=None, help="etcd endpoints")

    parser.add_argument(
        "--job_id", type=str, default=None, help="The identify id of this job")

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logging level, default is logging.INFO")

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log",
        help="The path for each process's log.If it's not set, the log will printed to default pipe."
    )

    parser.add_argument(
        "--hdfs_name",
        type=str,
        default=None,
        help="The hdfs_name used for edl.")

    parser.add_argument(
        "--hdfs_ugi",
        type=str,
        default=None,
        help="The hdfs_ugi used for edl.")

    # checkpoint will saved here
    parser.add_argument(
        "--hdfs_path",
        type=str,
        default=None,
        help="The hdfs_path used for edl.")

    #positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    #rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def _convert_args_to_dict(args):
    d = {}
    for k, v in six.iteritems(vars(args)):
        if v is not None:
            d[k] = v
    return d


def edl_barrier(job_env, pod, timeout):
    start = time.time()

    log_time = time.time()
    while True:
        try:
            db = get_global_etcd()
            leader = db.get_pod_leader()
            if leader is None:
                raise EdlGenerateClusterError("can't get leader")

            c = PodServerClient(leader.endpoint)
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
            logger.fatal(message)
            #raise EdlBarrierError(message)

        time.sleep(3)

    return None


def prepare(args):
    args_dict = _convert_args_to_dict(args)

    # job enviroment.
    job_env = JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))

    # get global etcd and lock
    db = get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    last_status = db.get_job_status()
    if last_status == Status.SUCCEED:
        logger.info("job:{} has completed! Need't try!".format(job_env.job_id))
        sys.exit(0)

    # local pod, and the pod's id does't change.
    pod = Pod()
    pod.from_env(job_env)

    # update pod status
    db.set_pod_status(pod.get_id(), Status.INITIAL)

    # launch pod server
    pod_server = None
    pod_server = PodServer(job_env, pod.get_id())
    pod_server.start(job_env, pod)
    logger.info("pod server started:[{}]".format(pod))

    return job_env, pod


def job_exit(leader_register,
             resource_register,
             watcher,
             pod,
             trainer_flag,
             register_flag,
             barrier_flag,
             resource_flag,
             timeout=300):
    local_flag = trainer_flag & register_flag & barrier_flag
    db = get_global_etcd()
    db.set_pod_flag(pod.get_id(), local_flag)

    begin = time.time()
    while True:
        try:
            if leader_register.is_leader():
                if db.wait_resource(cluster, timeout=15):
                    job_flag = trainer_flag & register_flag & barrier_flag & resource_flag
                    db.set_job_flag(job_flag)
                    logger.info("set job status:{} ok!".format(job_flag))
                    break
                raise EdlWaitFollowersReleaseError("can't wait resource")
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
    job_env, pod = prepare(args)

    # register pod resource to tell others:
    # this resource can use to train
    resource_register = PodResourceRegister(job_env, pod)

    # seize the leader
    leader_register = LeaderRegister(job_env, pod.get_id())

    # register rank and watch the rank
    # if the rank changed, the pods should restart the training proc.
    # pod exit if barrier error
    cluster = edl_barrier(job_env, pod, timeout=600)

    # update pod status
    db = get_global_etcd()
    db.set_pod_status(pod.get_id(), Status.RUNNING)

    # watcher after barrier
    watcher = Watcher(job_env, cluster, pod)

    procs = start_local_trainers(
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
        alive, trainer_flag = watch_local_trainers(procs, pod.trainers_num)
        if not alive or not trainer_flag:
            break

        if resource_register.is_stopped() or leader_register.is_stopped():
            terminate_local_procs()
            register_flag = False
            break

        # check job status second
        if watcher.changed:
            new_cluster = edl_barrier(job_env, pod, timeout=60)
            if not new_cluster:
                barrier_flag = False
                break

            terminate_local_procs(procs)

            cluster = new_cluster
            watcher = Watcher(job_env, cluster, pod)

            procs = start_local_trainers(
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
        leader_register=leader_register,
        resource_register=resource_register,
        watcher=watcher,
        pod=pod,
        trainer_flag=trainer_flag,
        register_flag=register_flag,
        barrier_flag=barrier_flag)


def run_commandline():
    utils.get_logger(log_level=10)
    args = _parse_args()
    launch(args)


if __name__ == '__main__':
    run_commandline()
