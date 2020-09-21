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


def _compare_two_list(list1, list2):
    if not isinstance(list2, list):
        return False

    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        if list1[i] != list2[i]:
            return False

    return True


def _compare_two_dict(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for k, v in six.iteritems(dict1):
        if k not in dict2:
            return False

        if isinstance(v, dict):
            if not _compare_two_dict(v, dict2[k]):
                return False
        elif isinstance(v, list):
            if not _compare_two_dict(v, dict2[k]):
                return False
        else:
            if v != dict2[k]:
                return False

    return True


class Serializable(SerializableBase):
    def to_json(self, filter_names=set()):
        d = {}
        for k, v in six.iteritems(self.__dict__):
            if k in filter_names:
                continue

            if isinstance(v, SerializableBase):
                d[k] = v.to_json()
                continue

            d[k] = v

        return json.dumps(d)

    def from_json(self, json_str):
        d = json.loads(json_str)
        for k, v in six.iteritems(self.__dict__):
            if k not in d:
                continue

            self.__dict__[k] = d[k]

        return self

    def __eq__(self, other):
        if other is None:
            return False

        return _compare_two_dict(self.__dict__, other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_json()
