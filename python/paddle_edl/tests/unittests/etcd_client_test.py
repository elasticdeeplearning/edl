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

import unittest
from paddle_edl.discovery.etcd_client import EtcdClient
import time
import threading
from etcd3.events import PutEvent, DeleteEvent


class TestEtcd(unittest.TestCase):
    def setUp(self):
        self.etcd = EtcdClient()
        self.etcd.init()

    def add(self):
        local_servers = {"127.0.0.1:1": "first", "127.0.0.1:2": "second"}

        for k, v in local_servers.items():
            self.etcd.set_server("job_1", k, v)

        servers = self.etcd.get_service("job_1")
        assert len(servers) == 2, "must two servers"

        for k, v in servers:
            value = local_servers[k]
            assert value == v

    def refresh(self):
        self.etcd.refresh("job_1", "127.0.0.1:1")
        self.etcd.refresh("job_1", "127.0.0.1:2")

    def get_service(self):
        servers = self.etcd.get_service("job_1")
        assert len(servers) == 0, "key must not alive when expired."
        self.etcd.refresh("job_1", "127.0.0.1:1")

    def remove_service(self):
        self.etcd.remove_service("job_1")
        servers = self.etcd.get_service("job_1")
        assert len(servers) == 0, "key must not alive after sevice is deleted."

    def test_etcd(self):
        self.add()
        self.etcd.remove_service("job_1")
        self.add()
        self.refresh()
        time.sleep(10)
        self.get_service()

    def update_key(self):
        self.etcd.set_server("job_2", "127.0.0.1:1", "first")

    def test_watch(self):
        events = []

        def watch_call_back(response):
            try:
                events.extend(response.events)
                for e in events:
                    key = EtcdClient.get_server_name_from_full_path(e.key,
                                                                    "job_2")
                    if type(e) == PutEvent:
                        print("put event key:{} value:{}".format(key, e.value))
                    elif type(e) == DeleteEvent:
                        print("delete event key:{} value:{}".format(key,
                                                                    e.value))
            except Exception as e:
                print("events len 1:", len(events))
                print(e)

        watch_id = self.etcd.watch_service("job_2", watch_call_back)

        t = threading.Thread(name="update_key_prefix", target=self.update_key)
        t.start()
        t.join()

        print("watch_id:", watch_id)
        time.sleep(3)
        self.etcd.cancel_watch(watch_id)

        print("events len:", len(events))
        assert len(events) == 1
        assert EtcdClient.get_server_name_from_full_path(
            events[0].key, "job_2") == '127.0.0.1:1'
        assert events[0].value == 'first'

    def test_lease(self):
        self.add()
        for i in range(20):
            self.refresh()
            servers = self.etcd.get_service("job_1")
            assert len(servers) == 2, "must two servers"
            time.sleep(2)


if __name__ == '__main__':
    unittest.main()
