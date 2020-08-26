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
from ..utils.pod import Pod, JobStatus
from ..utils.register import LeaderRegister, PodResourceRegister
from ..utils.etcd_db import EtcdDB as db
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


def edl_barrier(job_env, pod, leader_register, timeout):
    start = time.time()

    log_time = time.time()
    while True:
        try:
            leader = db.get_pod_leader()
            c = PodServerClient(leader.endpoint)

            cluster = c.barrier(job_env.job_id, pod.get_id())
            rank_register.update_pod_stage(pod)
        except Exception as e:
            if time.time() - log_time > 30:
                logger.info("wait to barrier now!")
                log_time = time.time()
            logger.debug("barrier error:{} {}".format(e,
                                                      traceback.format_exc()))

        if time.time() - start > timeout:
            message = "wait to barrier with all error:{} leader:[{}] current pod:[{}]".format(
                traceback.format_exc(), leader, pod)
            raise EdlBarrierError(message)

        time.sleep(3)


def on_rank_pods_changed(job_env,
                         cluster,
                         pod,
                         leader_register,
                         watcher,
                         timeout=600):
    """
    return new_cluster, job_status
    """
    start = time.time()
    while True:
        try:
            leader = db.get_pod_leader()
            c = PodServerClient(leader.endpoint)

            # barrier and get the changed pods.
            succeed, failed, added, inited = c.get_diff_pods(cluster)
            logger.info("succeed:{} failed:{} added:{}".format(succeed, failed,
                                                               added))
            if len(failed) == 0 and len(added) == 0:
                return None, True

            # all pods stop watch
            watcher.stop()

            # leader need't to regist
            if leader_register.is_leader():
                logger.info("leader need not to re-regist")
            else:
                leader_register.stop()
                # wait followers release their registers
                db.wait_following_ranks(added, timeout=20)

                # re-regist to reserver dense rank order
                leader_register = LeaderRegister(job_env, pod)
                logger.info("pod re-regist:{}".format(pod))

            new_cluster = edl_barrier(job_env, pod, timeout=60)

            # watch agagin
            watcher = Watcher(job_env.etcd_endpoints, job_env.job_id, pod,
                              cluster)
            watcher.start()

            return new_cluster, True
        except Exception as e:
            if time.time() - start >= timeout:
                logger.Fatal("on_ranks_pods_changed meets {}".format(
                    traceback.format_exc()))
                return None, False

            logger.debug("on_rank_pods_changed meets {}".format(
                traceback.format_exc()))
            time.sleep(3)
            continue

    return None, False


def set_pod_complete_flag(pod, local_status):
    # set pod's status
    if not local_status:
        db.set_pod_status(pod.get_id(), JobStatus.ERROR)
        logger.fatal("local trainers meets error!")
        return

    db.set_pod_status(pod.get_id(), JobStatus.COMPLETE)
    logger.info("local trainers succeeded!")


def set_job_complete_flag(leader_register,
                          local_status,
                          job_status,
                          timeout=60):
    start = time.time()

    while True:
        if leader_register.is_leader():
            return

        try:
            # wait all followers exit
            db.wait_following_ranks(timeout=10)

            if local_status and job_status:
                db.set_job_status(JobStatus.COMPLETE)
                logger.info("Congratulate! This job complete!")
        except:
            if time.time() - start >= timeout:
                return

        time.sleep(3)


def watch_registers(leader_register, resource_register):
    if leader_register.is_stoped() or resource_register.is_stoped():
        return False

    return True


def prepare(args):
    args_dict = _convert_args_to_dict(args)

    # job enviroment.
    job_env = JobEnv(args_dict)
    logger.info("get job env:{}".format(str(job_env)))

    # get global etcd and lock
    get_global_etcd(job_env.etcd_endpoints, job_env.job_id)

    last_status = db.get_job_status()
    if last_status is not None and last_status == JobStatus.COMPLETE:
        logger.info("job:{} has completed! Need't try!".format(job_env.job_id,
                                                               last_status))
        sys.exit(0)

    # local pod, and the pod's id does't change.
    pod = Pod()
    pod.from_env(job_env)

    # update pod status
    db.set_pod_status(pod.get_id(), JobStatus.INITIAL)

    # launch pod server
    pod_server = None
    pod_server = PodServer(job_env, pod.get_id())
    pod_server.start(job_env, pod)
    logger.info("pod server started:[{}]".format(pod))

    return job_env, pod


def launch(args):
    job_env, pod = prepare(args)

    # register pod resource to tell others:
    # this resource can use to train
    resource_register = PodResourceRegister(job_env.etcd_endpoints,
                                            job_env.job_id, pod)

    # seize the leader
    leader_register = LeaderRegister(job_env, pod)

    # register rank and watch the rank
    # if the rank changed, the pods should restart the training proc.
    # pod exit if barrier error
    cluster = edl_barrier(job_env, pod, timeout=600)

    # update pod status
    db.set_pod_status(pod.get_id(), JobStatus.RUNNING)

    # watcher after barrier
    watcher = Watcher(
        job_env.etcd_endpoints, job_env.job_id, pod, cluster=cluster)

    procs = start_local_trainers(
        cluster,
        pod,
        args.training_script,
        args.training_script_args,
        log_dir=args.log_dir)

    local_status = True
    job_status = True
    register_status = True
    while True:
        # check local status first
        alive, local_status = watch_local_trainers(procs, pod.trainers_num)
        if not alive or not local_status:
            break

        if resource_register.is_stopped() or leader_register.is_stopped():
            register_status = False
            break

        # check job status second
        if watcher.changed:
            new_cluster = edl_barrier(job_env, cluster, pod, leader_register,
                                      watcher)

            if new_cluster is None:
                time.sleep(2)
                continue

            # terminate and restart again
            terminate_local_procs(procs)

            if not job_status:
                break

            cluster = new_cluster
            procs = start_local_trainers(
                cluster,
                pod,
                args.training_script,
                args.training_script_args,
                log_dir=args.log_dir)

        time.sleep(2)

    # release resource in inverse order
    watcher.stop()

    # disappeared
    if not register_status:
        logger.fatal("register meets error and local exit!")
    else:
        # update pod status
        set_pod_complete_flag(pod, local_status)
        set_job_complete_flag(leader_register, local_status, job_status)

    leader_register.stop()
    resource_register.stop()


def run_commandline():
    utils.get_logger(log_level=10)
    args = _parse_args()
    launch(args)


if __name__ == '__main__':
    run_commandline()
