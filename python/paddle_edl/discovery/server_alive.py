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

import socket
from contextlib import closing


def is_server_alive(server):
    """ is server alive
    return alive, client_addr
    """
    alive = True
    client_addr = None
    ip, port = server.split(":")
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.settimeout(1.5)
            s.connect((ip, int(port)))
            client_addr = s.getsockname()
            s.shutdown(socket.SHUT_RDWR)
        except socket.error:
            alive = False
        return alive, client_addr
