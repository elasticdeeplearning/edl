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

import os
from paddle.distributed.fs_wrapper import BDFS


def download_hdfs_file(model_name, dst_path):
    """
    teacher model name
    dst_path: dst directory name
    """
    hdfs_name = os.getenv("PADDLE_DISTILL_HDFS_NAME")
    hdfs_ugi = os.getenv("PADDLE_DISTILL_HDFS_UGI")
    hdfs_path = os.getenv("PADDLE_DISTILL_HDFS_PATH")
    assert hdfs_name, "hdfs_name must be set"
    assert hdfs_ugi, "hdfs_ugi must be set"
    assert hdfs_path, "hdfs_path must be set"

    fs = BDFS(hdfs_name, hdfs_ugi)

    proto_path = hdfs_path + "/" + model_name + "/serving_server_conf.prototxt"
    fs.download(proto_path, dst_path)
