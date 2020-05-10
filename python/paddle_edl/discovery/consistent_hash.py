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


class ConsistentHash(object):
    @staticmethod
    def _get_slot(key):
        return int(hashlib.md5(key).hexdigest(), 16)

    def __init__(self, nodes, virtual_num=300):
        # NOTE. need deepcopy, otherwise outside maybe change
        self._nodes = copy.deepcopy(nodes)
        self._virtual_num = virtual_num

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

    def get_node(self, key):
        if len(self._nodes) == 0:
            return None

        slot = self._get_slot(key)
        # Find a virtual node >= key along the hash ring
        index = bisect.bisect_left(self._sorted_slots, slot)
        if index == len(self._sorted_slots):
            index = 0

        return self._slot_to_node[self._sorted_slots[index]]
