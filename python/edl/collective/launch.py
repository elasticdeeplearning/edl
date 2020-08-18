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

from ..utils.edl_env import JobEnv
from ..utils.cluster import Pod
from ..utils.register import PodRegister, PodResourceRegister, ETCD_POD_RANK, ETCD_POD_RESOURCE
from ..utils.watcher import Watcher, get_pod_leader
from ..utils.pod_server import PodServer
from ..utils.utils import logger
from ..utils import utils
from ..utils.global_vars import *


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

    # regist and get rank, leader is in it.
    # and leader will change the stage to a unique string
    register = PodRegister(job_env, pod)

    # all pods barrier on leader
    while True:
        try:
            leader = get_pod_leader()

            c = Client(leader.endpoint)
            c.Barrier(job_env.job_id, pod.id)
            break
        except Exception as e:
            logger.warning(
                "wait to barrier with all, context:job_env{} pod:{}".format(
                    job_env, pod))
            time.sleep(1)

        if time.time() - start > timeout:
            message = "can't barrier with all, context:job_env{} pod:{}".format(
                job_env, pod)
            logger.warning(message)
            raise EdlBarrierError(message)

    # watcher exit when cluster changed
    watcher = Watcher(job_env.etcd_endpoints, job_env.job_id)
    return register, watcher


def launch(args):
    args_dict = _convert_args_to_dict(args)
    job_env = JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))
    pod = Pod()
    pod.from_env(job_env)

    # get global etcd and lock
    get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    pod_server = PodServer()
    # port changed in it.
    pod_server.start(job_env, pod)
    logger.info("pod server started:[{}]".format(pod))

    # register pod resource, they can't be stopped.
    pod_rr = PodResourceRegister(job_env.etcd_endpoints, job_env.job_id, pod)

    # register rank and watch the rank
    # if the rank changed, the pods should restart the training proc.
    register, watcher = edl_barrier(job_env, pod, timeout=600)

    while True:
        cluster = watcher.get_cluster()
        procs = start_local_trainers(
            cluster,
            pod,
            args.training_script,
            args.training_script_args,
            log_dir=args.log_dir)

        if watcher.is_changed():
            logger.info("Cluster changed. New cluster:{}. Old Cluster:{}".
                        format(cluster2, cluster))

            register.stop()
            watcher.stop()
            terminiate_local_trainers(procs)
            # wait all pods info diappeared from etcd
            # don't change this time,since the ttl is set to 10s in registers
            # FIXME(gongwb): any other method?
            time.sleep(15)

            # barrier again
            register, watcher = edl_barrier(job_env, pod, timeout=600)
            continue

        alive = watch_local_trainers(procs, pod.trainers_num())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(1)

    register.complete()


def run_commandline():
    utils.get_logger(log_level=20)
    args = _parse_args()
    launch(args)


if __name__ == '__main__':
    run_commandline()
