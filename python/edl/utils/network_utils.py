import socket

from contextlib import closing

def get_extern_ip():
    return socket.gethostbyname(socket.gethostname())

def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def find_free_ports(num):
    if num <= 0:
        return None

    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None



