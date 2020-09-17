from edl.utils import constants
import json
from edl.utils.log_utils import logger

class Status(IntEnum):
    INITIAL = 0
    RUNNING = 1
    PENDING = 2
    SUCCEED = 3
    FAILED = 4

    @staticmethod
    def bool_to_status(b):
        if b:
            return Status.SUCCEED

        return Status.FAILED


def load_job_status_from_etcd(etcd):
    service = constants.ETCD_JOB_STATUS
    servers = etcd.get_service(service)

    assert len(servers) <= 1
    if len(servers) < 1:
        return None

    s = servers[0]
    d = json.loads(s.info)
    return d["status"]

def save_job_status_to_etcd(etcd, status):
    service = constants.ETCD_JOB_STATUS
    server = "status"
    info = json.dumps({"status": int(status)})
    etcd.set_server_permanent(service, server, info)

def save_job_flag_to_etcd(self, pod_id, flag):
    if flag:
        save_job_status_to_etcd(pod_id, constants.Status.SUCCEED)
        logger.info("This job succeeded!")
        return

    logger.fatal("This job meets error!")

def save_pod_status_to_etcd(etcd, pod_id, status):
    service = constants.ETCD_POD_STATUS
    server = pod_id
    info = json.dumps({"status": int(status)})

    etcd.set_server_permanent(service, server, info)

def load_pods_status_from_etcd(etcd):
    service = constants.ETCD_POD_STATUS
    servers = etcd.get_service(service)

    inited = set()
    running = set()
    succeed = set()
    failed = set()
    for server in servers:
        d = json.loads(server.info)
        status = d["status"]
        if status == int(constants.Status.FAILED):
            failed.add(server.server)
        elif status == int(constants.Status.SUCCEED):
            succeed.add(server.server)
        elif status == int(constants.Status.INITIAL):
            inited.add(server.server)
        elif status == int(constants.Status.RUNNING):
            running.add(server.server)

    return inited, running, succeed, failed
