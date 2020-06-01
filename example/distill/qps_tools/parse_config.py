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
from paddle_serving_client import Client as ServingClient


def get_ins_predicts(conf_file=None):
    """ May deprecated in future"""
    client_types = ['int64', 'float32']

    if conf_file is not None and os.path.isfile(conf_file):
        conf_file = conf_file
    elif os.path.isfile('./serving_conf/serving_client_conf.prototxt'):
        conf_file = './serving_conf/serving_client_conf.prototxt'
    else:
        conf_file = os.getenv('PADDLE_DISTILL_CONF_FILE')
        assert conf_file is not None
        assert os.path.isfile(conf_file)

    client = ServingClient()
    client.load_client_config(conf_file)

    feeds = client.get_feed_names()
    feeds_shapes = []
    feeds_dtype = []
    for feed_name in feeds:
        shape = client.feed_shapes_[feed_name]
        feeds_shapes.append(tuple(shape))
        feeds_dtype.append(client_types[client.feed_types_[feed_name]])

    predicts = client.get_fetch_names()
    return feeds, feeds_shapes, feeds_dtype, predicts
