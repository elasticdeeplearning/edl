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

from google.protobuf import text_format
from paddle_serving_client import m_config
from .shared_data import Var

_serving_type_to_dtype = ['int64', 'float32']


def _serving_var_to_var(var):
    name = var.alias_name
    try:
        stype = var.feed_type
    except AttributeError:
        stype = var.fetch_type
    dtype = _serving_type_to_dtype[stype]
    shape = tuple(var.shape)
    return Var(name, dtype, shape)


def parse_serving_conf(path):
    """ parse paddle serving client config file
    :return feed_vars, fetch_vars
    """
    model_conf = m_config.GeneralModelConfig()
    f = open(path, 'r')
    model_conf = text_format.Merge(str(f.read()), model_conf)

    feed_vars = dict()
    fetch_vars = dict()

    for var in model_conf.feed_var:
        svar = _serving_var_to_var(var)
        feed_vars[svar.name] = svar

    for var in model_conf.fetch_var:
        svar = _serving_var_to_var(var)
        fetch_vars[svar.name] = svar

    return feed_vars, fetch_vars
