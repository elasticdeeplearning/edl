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
    if len(dict1) != len(dict2):
        print("len(dict1) != len(dict2)", dict1, dict2)
        return False

    for k, v in six.iteritems(dict1):
        if k not in dict2:
            print("k not in dict2", k, dict1[k], type(dict1[k]))
            print("dict2:", dict2)
            return False

        if isinstance(v, dict):
            if not _compare_two_dict(v, dict2[k]):
                return False
        else:
            if v != dict2[k]:
                print("k not equal:", k, v, dict2[k])
                return False

    return True


class Serializable(SerializableBase):
    def _not_contain_custom_cls(self, dict_data):
        for k, v in six.iteritems(dict_data):
            if isinstance(v, SerializableBase):
                return False

        return True

    def _dict_to_json(self, dict_data, filter_names=set()):
        d = {}
        for k, v in six.iteritems(dict_data):
            if k in filter_names:
                continue

            if isinstance(v, SerializableBase):
                d[k] = v.to_json()
            elif isinstance(v, dict):
                if self._not_contain_custom_cls(v):
                    d[k] = v
                    continue

                d[k] = self._dict_to_json(v)
            else:
                d[k] = v

        return json.dumps(d)

    def to_json(self, filter_names=set()):
        json_str = self._dict_to_json(self.__dict__, filter_names=filter_names)
        return json_str

    def __eq__(self, other):
        if other is None:
            return False

        return _compare_two_dict(self.__dict__, other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_json()
