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

import copy
import unittest
import six
from edl.discovery.consistent_hash import ConsistentHash


class TestConsistentHash(unittest.TestCase):
    def test_consistent_hash(self):
        nodes = ['127.0.0.1:1234', '127.0.0.1:2345', '127.0.0.1:3456']
        sample_count = 10000
        node_to_count = {key: 0 for key in nodes}
        sample_to_node = dict()

        cs_hash = ConsistentHash(nodes)

        def hash_test(ip):
            for i in range(sample_count):
                key = '{}:{}'.format(ip, i)
                node = cs_hash.get_node(key)
                if key not in sample_to_node:
                    sample_to_node[key] = node
                    node_to_count[node] += 1
                else:
                    old_node = sample_to_node[key]
                    node_to_count[old_node] -= 1

                    sample_to_node[key] = node
                    node_to_count[node] += 1

            for node, count in six.iteritems(node_to_count):
                print('node={}, count={}'.format(node, count))

        hash_test('1.1.1.1')
        old_node_to_count = copy.deepcopy(node_to_count)
        for count in node_to_count.values():
            # test Balance
            assert count > 3000

        # remove node
        print('\nremove node={}'.format(nodes[1]))

        cs_hash.remove_node(nodes[1])
        hash_test('1.1.1.1')
        # test Monotonicity, remove
        assert 0 == node_to_count[nodes[1]]

        # recover node
        print('\nrecover node={}'.format(nodes[1]))
        cs_hash.add_new_node(nodes[1])
        hash_test('1.1.1.1')
        # test Monotonicity, recover
        assert node_to_count == old_node_to_count

        # add new node
        new_node = '8.8.8.8:8888'
        print('\nadd new node={}'.format(new_node))
        nodes.append(new_node)
        node_to_count[new_node] = 0
        cs_hash.add_new_node(new_node)

        hash_test('8.8.8.8')
        # test Balance, Monotonicity
        assert node_to_count[new_node] < 3000


if __name__ == '__main__':
    unittest.main()
