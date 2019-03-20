#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from pprint import pprint
import time

JOB_STATUS_NOT_EXISTS = 0
JOB_STATUS_PENDING = 1
JOB_STATUS_RUNNING = 2
JOB_STATUS_FINISHED = 3
JOB_STSTUS_KILLED = 4


class JobInfo(object):
    def __init__(self, name):
        self.name = name
        self.started = False
        self.status = JOB_STATUS_NOT_EXISTS
        self.submit_time = -1
        self.start_time = -1
        self.end_time = -1
        self.parallelism = 0
        self.cpu_utils = ''

    def status_str(self):
        if self.status == JOB_STATUS_FINISHED:
            return 'FINISH'
        elif self.status == JOB_STATUS_PENDING:
            return 'PENDING'
        elif self.status == JOB_STATUS_NOT_EXISTS:
            return 'N/A'
        elif self.status == JOB_STATUS_RUNNING:
            return 'RUNNING'
        elif self.status == JOB_STSTUS_KILLED:
            return 'KILLED'


class Collector(object):
    '''
    Collector monitor data from Kubernetes API
    '''

    def __init__(self):
        config.load_kube_config()
        self.namespace = config.list_kube_config_contexts()[1]['context'].get("namespace", "default")
        self.cpu_allocatable = 0
        self.gpu_allocatable = 0
        self.cpu_requests = 0
        self.gpu_requests = 0
        self._namespaced_pods = []
        # Collect cluster wide resource
        self._init_allocatable()

        self._pods = []

        # collect master, pserver and trainer phase for each training-job
        self._job_phases = {}

    def _init_allocatable(self):
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.list_node()
            cpu = 0
            gpu = 0
            for item in api_response.items:
                allocate = item.status.allocatable
                cpu += int(allocate.get('cpu', 0))
                gpu += int(allocate.get('gpu', 0))
            self.cpu_allocatable = cpu
            self.gpu_allocatable = gpu
        except ApiException as e:
            print("Exception when calling CoreV1Api->list_node: %s\n" % e)

    def _real_cpu(self, cpu):
        if cpu:
            if cpu.endswith('m'):
                return 0.001 * int(cpu[:-1])
            else:
                return int(cpu)
        return 0

    def _once_job_phases(self, pods):
        # job_phases records the master, pserver and trainer
        # phase for each training-job:
        # {
        #   <job-name>:
        #     {
        #       "master":  [],
        #       "pserver": [],
        #       "trainer": [] 
        #     }
        # }
        self._job_phases = {}
        for item in pods:
            if not item.metadata.labels:
                continue
            for k, v in item.metadata.labels.items():
                if k.startswith("paddle-job") and v not in self._job_phases:
                    self._job_phases.update({v:{"master": list(), "pserver": list(), "trainer": list()}})
                if k == "paddle-job-master":
                    self._job_phases[v]["master"].append(item.status.phase)             
                elif k == "paddle-job-pserver":
                    self._job_phases[v]["pserver"].append(item.status.phase)             
                elif k == "paddle-job":
                    self._job_phases[v]["trainer"].append(item.status.phase)

    def run_once(self):
        api_instance = client.CoreV1Api()
        config.list_kube_config_contexts()
        try:
            api_response = api_instance.list_pod_for_all_namespaces()
            self._pods = api_response.items
            self._namespaced_pods = []
            for pod in self._pods:
                if pod.metadata.namespace == self.namespace:
                    self._namespaced_pods.append(pod)
            self._once_job_phases(self._namespaced_pods)
        except ApiException as e:
            print(
                "Exception when calling CoreV1Api->list_pod_for_all_namespaces: %s\n"
                % e)
        return int(time.time())

    def get_running_trainers(self):
        job_running_trainers = dict()
        for k, v in self._job_phases.items():
            if k not in job_running_trainers:
                job_running_trainers[k] = 0

            if "Pending" in v["pserver"] or \
                "Pending" in v["master"]:
                continue
            cnt = 0 
            for p in v["trainer"]:
                if p == "Running":
                    cnt += 1            
            job_running_trainers[k] = cnt
        if job_running_trainers:
            return "|".join(["%s:%d" % (k, v) for k, v in job_running_trainers.items()])
        else:
            return "-"

    def cpu_utils(self):
        cpu = 0
        for item in self._pods:
            if item.status.phase != 'Running':
                continue
            for container in item.spec.containers:
                requests = container.resources.requests
                if requests:
                    cpu += self._real_cpu(requests.get('cpu', None))

        return '%0.2f' % ((100.0 * cpu) / self.cpu_allocatable)

    def gpu_utils(self):
        gpu = 0
        for item in self._pods:
            if item.status.phase != 'Running':
                continue
            for container in item.spec.containers:
                limits = container.resources.limits
                if limits:
                    gpu += int(limits.get('alpha.kubernetes.io/nvidia-gpu', 0))
        if not self.gpu_allocatable:
            return '0'
        return '%0.2f' % ((100.0 * gpu) / self.gpu_allocatable)

    def get_paddle_pods(self):
        pods = []
        for item in self._namespaced_pods:
            if not item.metadata.labels:
                continue
            for k, v in item.metadata.labels.items():
                if k.startswith('paddle-job'):
                    pods.append((item.metadata.name, item.status.phase))
        return pods

    def get_submitted_jobs(self):
        return len(self._job_phases)

    def get_pending_jobs(self):
        cnt = 0
        for _, phases in self._job_phases.items():
            if "Pending" in phases["pserver"] or \
                "Pending" in phases["master"] or \
                len(phases["trainer"]) == 0 or \
                (len(set(phases["trainer"])) == 1 and "Pending" in phases["trainer"]):
                cnt += 1
        return cnt

    def get_running_pods(self, labels):
        pods = 0
        for item in self._namespaced_pods:
            if item.metadata.labels:
                for k, v in item.metadata.labels.items():
                    if k in labels and labels[k] == v and \
                            item.status.phase == 'Running':
                        pods += 1

        return pods

if __name__ == "__main__":
    c = Collector()
    print("SUBMITED-JOBS\tPENDING-JOBS\tRUNNING-TRAINERS\tCPU-UTILS")
    while True:
        c.run_once()
        print "\t".join([
          str(c.get_submitted_jobs()),
          str(c.get_pending_jobs()),
          c.get_running_trainers(),
          c.cpu_utils() + "%"
        ])
        time.sleep(10)