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
       python -m paddle.distributed.launch --selected_gpus="0,1,2,3" \
         your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            python -m paddle.distributed.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.16 \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            python -m paddle.distributed.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
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

from utils import *
import edl_utils
from http_store import kv_server


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
        description='''start paddle training using multi-process mode.
NOTE: your train program ***must*** run as distributed nccl2 mode,
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
And your train program must read environment variables below in order to let different
process init properly:
FLAGS_selected_gpus
PADDLE_TRAINER_ID
PADDLE_CURRENT_ENDPOINT
PADDLE_TRAINERS_NUM
PADDLE_TRAINER_ENDPOINTS
POD_IP (current node ip address, not needed for local training)
''')

    #Optional arguments for the launch helper
    parser.add_argument(
        "--cluster_node_ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
    parser.add_argument(
        "--node_ip",
        type=str,
        default="127.0.0.1",
        help="The current node ip. ")
    parser.add_argument(
        "--use_paddlecloud",
        action='store_true',
        help="wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument."
    )
    parser.add_argument(
        "--started_port",
        type=int,
        default=None,
        help="The trainer's started port on a single node")

    parser.add_argument(
        "--print_config",
        type=bool,
        default=True,
        help="Print the config or not")

    parser.add_argument(
        "--selected_gpus",
        type=str,
        default=None,
        help="It's for gpu training and the training process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,  # logging.INFO, details are here:https://docs.python.org/3/library/logging.html#levels
        help="Logging level, default is logging.INFO")

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
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


def launch(args):
    cluster = None
    pod = None
    hdfs = None

    edl_env = edl_utils.Edlenv()
    assert edl_env.is_under_edl(), "edl launch must run under edl env"

    hdfs = edl_utils.get_hdfs_from_args(args)
    cluster, pod = edl_utils.edl_barrier(edl_env, hdfs, timeout=15 * 60)
    logger.info("get cluster from edl:{}".format(cluster))

    procs = start_local_trainers(
        cluster,
        pod,
        args.training_script,
        args.training_script_args,
        log_dir=args.log_dir)

    while True:
        cluster2, pod = edl_env.get_cluster(hdfs)

        if cluster2 != cluster:
            logger.info("Cluster changed. New cluster:{}. Old Cluster:{}".
                        format(cluster2, cluster))
            terminate_local_procs(procs)

            cluster, pod = edl_utils.edl_barrier(
                edl_env, hdfs, timeout=30 * 60)

            procs = start_local_trainers(
                cluster,
                pod,
                args.training_script,
                args.training_script_args,
                log_dir=args.log_dir)

        alive = watch_local_trainers(procs, cluster.trainers_nranks())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(3)

    edl_utils.edl_barrier(edl_env, hdfs)


if __name__ == "__main__":
    args = _parse_args()

    logger = get_logger(args.log_level)

    if args.print_config:
        _print_arguments(args)

    launch(args)
