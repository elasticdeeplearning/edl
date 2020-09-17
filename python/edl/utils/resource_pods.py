from edl.utils import constants
from edl.utils import pod
from edl.utils import error_utils

def get_resource_pods_dict(etcd):
    servers = etcd.get_service(constants.ETCD_POD_RESOURCE)

    pods = {}
    for s in servers:
        p = pod.Pod()
        p.from_json(s.info)
        pods[p.get_id()] = p

    return pods

@error_utils.handle_errors_until_timeout
def wait_resource(self, pod, timeout=15):
    pods = get_resource_pods_dict()
    if len(pods) == 1:
        if pod.get_id() in pods:
            return True

    if len(pods) == 0:
        return True

    return False
