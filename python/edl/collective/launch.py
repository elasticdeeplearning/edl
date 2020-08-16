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

from paddle_edl.utils.utils import *
from edl_env import JobEnv
from .cluster import Pod
from .register import PodRegister
from .watcher import Watcher
import paddle_edl.utils.master_client as master_client


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


def edl_barrier(job_env, pod, timeout=60):
    # regist and get rank
    register = PodRegister(job_env, pod)

    # watcher
    watcher = Watcher(job_env.etcd_endpoints, job_env.job_id)
    watcher.watch()

    # cluster must have mnimum pods
    cluster = None
    start = time.time()
    while True:
        cluster = watcher.get_cluster()
        if len(cluster.pods) < job_env.min_nodes:
            time.sleep(1)

            timeout -= 1
            if timeout <= 0:
                logger.warning("can't get enough nodes_num:{}, nodes:{}".
                               format(len(cluster.pods), cluster))
            continue

    # all pods barrier on master to avoid pod from last job stage
    master = cluster.pods[0]
    endpoint = "{}:{}".format(master.addr, master.port)
    start = time.time()
    while True:
        c = Client(endpoint)
        try:
            if not c.Barrier(job_env.job_id, pod.id):
                continue
        except Exception as e:
            time.sleep(1)
            if time.time() - start > timeout:
                logger.warning("can't barrier of nodes_num:{} nodes:{}".format(
                    len(cluster.pods), cluster))
            continue

    return register, watcher


def launch(args):
    job_env = JobEnv(args)
    pod = Pod()
    pod.from_env(job_env)

    pod_server = PodServer()
    # port changed in it.
    pod_server.start(jobe_env, pod)
    logger.info("pod server started:{}", pod)

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

            r.stop()
            w.stop()
            terminiate_local_trainers(procs)
            # wait all pods info displayed from etcd
            # don't change this time,since the ttl is set to 10s in registers
            # FIXME(gongwb): any other method?
            time.sleep(15)
            continue

        alive = watch_local_trainers(procs, pod.trainers_num())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(1)

    r.complete()
