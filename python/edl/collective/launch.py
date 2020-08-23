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
from ..utils.cluster import Pod, JobStatus
from ..utils.register import PodRankRegister, PodResourceRegister
from ..utils.edl_db import EtcdDB as db
from ..utils.watcher import Watcher, get_pod_leader, get_cluster
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
    """
    pod under resource barrier togather
    """
    start = time.time()

    leader = get_pod_leader()
    c = PodServerClient(leader.endpoint)
    log_time = time.time()
    # all pods barrier on leader
    while True:
        try:
            c.barrier(job_env.job_id, pod.get_id())
            break
        except Exception as e:
            if time.time() - log_time > 30:
                logger.info("wait to barrier now!")
                log_time = time.time()
            time.sleep(3)

        if time.time() - start > timeout:
            message = "wait to barrier with all error:{} leader:[{}] current pod:[{}]".format(
                traceback.format_exc(), leader, pod)
            raise EdlBarrierError(message)

    cluster = watcher.get_cluster()

    procs = start_local_trainers(
        cluster,
        pod,
        args.training_script,
        args.training_script_args,
        log_dir=args.log_dir)


def on_world_changed(job_env, pod, rank_register, watcher):
    """
    return if job failed.
    """
    logger.info("on_world_changed")

    # all pods stop watch
    watcher.stop()

    # terminate and restart again
    terminate_local_procs()

    # is resource enough?
    resource_ids = db.get_pods_ids_from_resource()
    if len(resource_ids) < job_env.min_nodes:
        logger.info("now pods resource:{}\
                is smaller than need:{}, job exit!"
                    .format(resource_ids, job_env.min_nodes))
        return False

    # leader need't to regist
    if rank_register.is_leader():
        logger.info("leader need not to re-regist")
    else:
        rank_register.stop()
        # wait followers release their registers
        db.wait_following_ranks()

        # re-regist to reserver dense rank order
        rank_register = PodRankRegister(job_env, pod)
        logger.info("pod re-regist:{}".format(pod))

    edl_barrier(job_env, pod, timeout=60)

    # watch agagin
    watcher = Watcher(job_env.etcd_endpoints, job_env.job_id, pod)


def launch(args):
    args_dict = _convert_args_to_dict(args)

    # job enviroment.
    job_env = JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))

    # get global etcd and lock
    get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    flag = get_job_complete_flag()
    if flag is not None:
        if flag:
            logger.info("job:{} has completed {}! Can't try!".format(
                job_env.job_id, flag))
            sys.exit(1)

    # local pod, and the pod's id does't change.
    pod = Pod()
    pod.from_env(job_env)

    # launch pod server
    pod_server = None
    pod_server = PodServer(pod.get_id())
    pod_server.start(job_env, pod)
    logger.info("pod server started:[{}]".format(pod))

    # register pod resource, they can't be stopped.
    resource_register = PodResourceRegister(job_env.etcd_endpoints,
                                            job_env.job_id, pod)

    # regist and get rank, leader is in it.
    # and leader will change the stage to a unique string
    rank_register = PodRankRegister(job_env, pod)

    # register rank and watch the rank
    # if the rank changed, the pods should restart the training proc.
    edl_barrier(job_env, pod, timeout=600)

    # watcher after barrier
    watcher = Watcher(job_env.etcd_endpoints, job_env.job_id, pod)

    cluster = watcher.get_cluster()
    procs = start_local_trainers(
        cluster,
        pod,
        args.training_script,
        args.training_script_args,
        log_dir=args.log_dir)

    local_status = True
    job_status = True
    while True:
        # check local status first
        alive, local_status = watch_local_trainers(procs, pod.trainers_num)
        if not alive or not local_status:
            break

        # check job status second
        if watcher.changed:
            job_status = on_world_changed(job_env, pod, rank_register, watcher,
                                          args)
            if not job_status:
                break

            cluster = watcher.get_cluster()
            procs = start_local_trainers(
                cluster,
                pod,
                args.training_script,
                args.training_script_args,
                log_dir=args.log_dir)

        time.sleep(2)

    # set pod's status
    if not local_status:
        logger.fatal("local trainers meets error!")
    else:
        logger.info("local trainers succeeded!")

    if rank_register.is_leader():
        if local_status and job_status:
            db.set_job_complete_flag(True)
            logger.info("Congratulate! This job complete!")

    # release resource in inverse order
    watcher.stop()
    rank_register.stop()
    resource_register.stop()


def run_commandline():
    utils.get_logger(log_level=10)
    args = _parse_args()
    launch(args)


if __name__ == '__main__':
    run_commandline()
