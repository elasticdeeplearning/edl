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

# Copyright 2015 gRPC authors.
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
"""Runs protoc with the gRPC plugin to generate messages and gRPC stubs."""

from grpc_tools import protoc
import pkg_resources
import os
import sys

print("run code gen python verion:", sys.version_info)

# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. data_server.proto
protoc.main((
    '',
    '-I.',
    '--python_out=.',
    '--grpc_python_out=.',
    'pod_server.proto', ))

protoc.main((
    '',
    '-I.',
    '--python_out=.',
    '--grpc_python_out=.',
    'data_server.proto', ))

proto_include = pkg_resources.resource_filename('grpc_tools', '_proto')
protoc.main((
    '',
    '-I.',
    '-I{}'.format(proto_include),
    '--python_out=.',
    '--grpc_python_out=.',
    'distill_discovery.proto', ))
