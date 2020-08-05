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
Usage:
    In both of single node training or multiple node training, this module 
launch a process on each of the given gpu card.
    1. for single node training with all visible gpu cards:
       python -m paddle.distributed.launch \
         your_training_py (arg1 arg2 and all others)
    
    2. for single node training with [0,4) cards
       python -m paddle_edl.collective.launch --selected_gpus="0,1,2,3" \
         your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            python -m paddle.collective.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.16 \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            python -m paddle.collective.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.17 \
                your_training_py (arg1 arg2 and all others)
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
from .watcher import MasterWatcher
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


def edl_barrier(master_dog, job_env, pod_env, timeout=15):
    c = master_client.Client(master_dog.get_master().endpoint)
    try:
        pb_cluster = c.barrier(job_env.job_id, pod_env.pod_id, timeout)
    except Exception as e:
        if type(e) is exception.PodDroppedError:
            logger.info("job_id:{} pod_id:{} was dropped".format(job_id,
                                                                 pod_id))
            sys.exit(0)

        logger.info("job_id:{} pod_id:{} barrier error:{}".format(
            job_id, pod_id, e.value))
        sys.exit(1)

    cluster = Cluster()
    cluster.init_from_pb(pb_cluster)

    pod = cluster.get_pod_by_id(pod_env.pod_id)
    return cluster, pod


# wait until master is set or timeout
def get_master(master_dog, timeout=300):  #s
    start = time.time()
    while True:
        if master_dog.get_master() is None:
            time.sleep(1)
            continue

    return master_dog.get_master()


def launch(args):
    job_env = JobEnv(args)
    pod_env = Pod()
    pod_env.init_from_env(job_env)

    # pod register
    pod_register = PodRegister(job_env, pod_env)

    # try to register master
    master_register = MasterRegister()

    # watch master
    master_watcher = MasterWatcher(job_env.etcd_endpoints, job_env.job_id)
    global_master = get_master(master_watcher)

    cluster, pod = edl_barrier(master_watcher, job_env, pod_env, None, 15 * 60)
    logger.info("get cluster from edl:{}".format(cluster))

    master_client = master_client.Client(master_watcher)
    while True:
        cluster2, pod = master_client.get_cluster()

        if cluster2.stage != cluster.stage:
            logger.info("Cluster changed. New cluster:{}. Old Cluster:{}".
                        format(cluster2, cluster))

            edl_process.terminiate(procs)

            cluster, pod = edl_barrier(
                job_env, pod_env, local_procs=procs, timeout=15 * 60)

            procs = edl_process.start_local_trainers(
                cluster,
                pod,
                args.training_script,
                args.training_script_args,
                log_dir=args.log_dir)

        alive = edl_process.watch_local_trainers(procs,
                                                 cluster.trainers_nranks())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(3)

    # normal exit
    pod_reg.complete()
