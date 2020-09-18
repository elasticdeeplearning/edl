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
import json
import six

class SerializableBase(object):
    def to_json(self):
        raise NotImplementedError

    def from_json(self):
        raise NotImplementedError

def _compare_two_dict(dict1, dict2):
    if len(dict1) != dict2:
        return False

    for k,v in six.iteritems(dict1):
        if k not in dict1:
            return False

        if isinstance(v, dict):
            if not _compare_two_dict(v, dict2[k]):
                return False
        else:
            if v != dict2[k]:
                return False

    return True

class Serializable(Base):
    def to_json(self):
        d = {}
        for k, v in six.iteritems(dict):
            if isinstance(v, SerializableBase):
                d[k] = v.to_json()
            else:
                d[k] = v

        return json.dumps(d)

    def from_json(self, json_str):
        d = json.loads(json_str)
        for k, v in self.__dict__:
            if k not in d:
                return

            if isinstance(v, Serializable):
                v.from_json(d[k])
                continue

            v = d[k]

    def __eq__(self, other):
        if other is None:
            return False

        return _compare_two_dict(self.__dict__, other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

