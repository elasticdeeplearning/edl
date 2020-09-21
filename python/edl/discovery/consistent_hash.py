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

import bisect
import copy
import hashlib
import sys


class _ConsistentHashData(object):
    def __init__(self, nodes, virtual_num=300):
        # NOTE. need deepcopy, otherwise outside maybe change
        self._nodes = copy.deepcopy(nodes)
        self._virtual_num = virtual_num
        self._version = 1

        # as c++ std::map
        self._slot_to_node = dict()  # hash slot to node
        self._sorted_slots = []  # virtual node sorted slots

        # init hash ring
        for node in self._nodes:
            self._add_node(node)
        self._sorted_slots.sort()  # NOTE. sort

    def _add_node(self, node):
        # NOTE. this func is not sort
        for i in range(self._virtual_num):
            vnode = '{}-v{}'.format(node, i)
            slot = self._get_slot(vnode)

            # Fix slot hash conflict, the probability
            # must be very very small!!!
            if slot in self._slot_to_node:
                old_node = self._slot_to_node[slot]
                if old_node <= node:  # select smaller
                    continue

            self._slot_to_node[slot] = node
            self._sorted_slots.append(slot)

    def add_new_node(self, node):
        if node in self._nodes:
            return
        self._nodes.append(node)

        self._add_node(node)
        self._sorted_slots.sort()  # NOTE. sort
        self._version += 1

    def remove_node(self, node):
        if node not in self._nodes:
            return
        self._nodes.remove(node)

        for i in range(self._virtual_num):
            vnode = '{}-v{}'.format(node, i)
            slot = self._get_slot(vnode)
            # NOTE. if hash conflict, may be another node
            if node == self._slot_to_node[slot]:
                del self._slot_to_node[slot]
                self._sorted_slots.remove(slot)

        self._version += 1

    def get_node(self, key):
        # return node
        if len(self._nodes) == 0:
            return None

        slot = self._get_slot(key)
        # Find a virtual node >= key along the hash ring
        index = bisect.bisect_left(self._sorted_slots, slot)
        if index == len(self._sorted_slots):
            index = 0

        return self._slot_to_node[self._sorted_slots[index]]

    def get_node_nodes(self, key):
        # return node, nodes, version
        if len(self._nodes) == 0:
            return None, self._nodes, self._version

        node = self.get_node(key)
        return node, self._nodes, self._version

    @staticmethod
    def _get_slot(key):
        if sys.version_info < (3, ):
            return int(hashlib.md5(key).hexdigest(), 16)
        else:
            return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)


class ConsistentHash(object):
    """ No lock write with 1 thread write, multi thread read.
    NOTE. Only 1 thread can write. I think it's enough.
    NOTE. The read data may not be up-to-date, but it doesn't matter.
    """

    def __init__(self, nodes, virtual_num=300):
        self._hash_data = _ConsistentHashData(nodes, virtual_num)

    def add_new_node(self, node):
        # no need add
        if node in self._hash_data._nodes:
            return

        new_hash_data = copy.deepcopy(self._hash_data)
        new_hash_data.add_new_node(node)
        self._hash_data = new_hash_data

    def remove_node(self, node):
        # no need remove
        if node not in self._hash_data._nodes:
            return

        new_hash_data = copy.deepcopy(self._hash_data)
        new_hash_data.remove_node(node)
        self._hash_data = new_hash_data

    def get_node(self, key):
        # return node
        hash_data = self._hash_data
        return hash_data.get_node(key)

    def get_node_nodes(self, key):
        # return node, nodes, version
        hash_data = self._hash_data
        return hash_data.get_node_nodes(key)
