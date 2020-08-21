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

import functools
import logging
import socket
import time
import os
import signal
import copy
import sys
import subprocess
from contextlib import closing
import socket
import psutil

from .utils import logger


class TrainerProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.rank = None
        self.cmd = None
        self.log_offset = None
        self.local_rank = None


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         log_dir=None):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = {
            "PADDLE_TRAINER_ID": "%d" % t.global_rank,  # global rank
            "PADDLE_TRAINER_RANK_IN_POD": "%d" % t.rank_in_pod,
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.get_trainers_world_size(),
            "PADDLE_TRAINER_ENDPOINTS":
            ",".join(cluster.get_trainers_endpoints()),
        }

        current_env.update(proc_env)

        #logger.debug("trainer proc env:{}".format(current_env))

        cmd = [sys.executable, "-u", training_script] + training_script_args

        logger.debug("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None
        if log_dir is not None:
            logger.debug("mkdir {}".format(log_dir))
            os.system("mkdir -p {}".format(log_dir))
            fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.global_rank
        tp.log_fn = fn
        tp.local_rank = idx
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    logger.info("all cluster trainers:{}".format(
        cluster.get_trainers_endpoints()))
    return procs


def terminate_local_procs(procs):
    decents = []
    for child in psutil.Process(os.getpid()).children(recursive=True):
        decents.append(child)

    for p in procs:
        p.log_fn.close()

    # try to kill
    for p in decents:
        p.send_signal(signal.SIGTERM)

    # wait
    gone, alive = psutil.wait_procs(decents, timeout=3)
    for p in alive:
        p.kill()

    # still alive?
    gone, alive = psutil.wait_procs(decents, timeout=1)
    if len(alive) != 0:
        logger.fatal("can't kill all process and exit")
        exit(1)

    logger.info("terminate all procs")


def watch_local_trainers(procs, nranks):
    """
    return alive_or_not, ok_or_not
    """
    try:
        alive = _watch_local_procs(procs, nranks)
    except Exception as e:
        return False, False

    return alive, True


def pull_worker_log(tp):
    if tp.log_fn:
        with open(tp.log_fn.name, 'r') as fin:
            fin.seek(tp.log_offset, 0)
            for line in fin:
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(
                        'UnicodeEncodeError occurs at this line. '
                        'Please refer to the original log file "%s"\n' %
                        tp.log_fn.name)
            tp.log_offset = fin.tell()


def watch_local_procs(procs, nranks):
    """
    If proc exit unnormally, this function will raise exception.
    """
    try:
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            ret = p.proc.poll()
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)

            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

        if error:
            raise
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_local_procs(procs)
        raise
    except SystemExit:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise
    except:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise

    return alive
