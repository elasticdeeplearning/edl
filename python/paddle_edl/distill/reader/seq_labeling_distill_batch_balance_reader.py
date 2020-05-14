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

from paddle_serving_client import Client
import sys
import random
import time
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import io
import signal
from bert_reader import BertReader, BertSequenceReader
import multiprocessing as mps
from threading import Thread
import paddle.fluid.incubate.data_generator as dg
from DistillServerBalance.client import Client as BClient


def dispatcher(input_queue, fin, endpoint2process):
    #for line in fin:
    #    input_queue.put(line)

    while True:
        with open('data/output_0.txt') as f:
            for line in f:
                input_queue.put(line)

    sys.stderr.write('All data reading completed\n')

    while not input_queue.empty():
        time.sleep(1)

    sys.stderr.write('All the data has been processed')

    for i in range(len(endpoint2process)):
        input_queue.put(None)


def client_producer(input_queue, outputqueue, endpoint, bert_reader,
                    stop_flag):
    # def exit_gracefully(signum, frame):
    #     sys.stderr.write("exit one service\n")
    #     os._exit(0)

    # signal.signal(signal.SIGTERM, exit_gracefully)

    #f = open(os.devnull, 'w')
    #f = open(str(endpoint) + '.log', 'a')
    #sys.stdout = f
    client = Client()
    client.load_client_config("ernie_crf_client/serving_client_conf.prototxt")
    client.connect([endpoint])

    batch_size = 8
    line_list = []
    feed_dict_list = []
    # Last transaction completed
    transaction_failed_count = 0

    while True:
        if stop_flag.value == 1:
            sys.stderr.write('start stop {} process\n'.format(endpoint))
            # if transaction failed, hold back data to the input_queue
            if transaction_failed_count != 0:
                for line in line_list:
                    input_queue.put(line)
            try:
                client.release()
            finally:
                sys.stderr.write('stopped {} process\n'.format(endpoint))
            return

        if transaction_failed_count == 0:
            try:
                line = input_queue.get(timeout=5)
                # line_list may not null? break
                if line == None:
                    break
            except:
                continue

            group = line.split("\t")
            feed_dict = bert_reader.process_words_and_labels(group[0],
                                                             group[1])
            labels = feed_dict["label_ids"]
            del feed_dict["label_ids"]
            unpad_seq_len = feed_dict["unpad_seq_lens"][0]
            if unpad_seq_len <= 0:
                continue
            del feed_dict["unpad_seq_lens"]

            line_list.append(line)
            feed_dict_list.append(feed_dict)
        elif len(feed_dict_list) != batch_size:
            # retry, but feed_dict_list is wrong
            sys.stderr.write('feed size error: feed={} != batch={}'.format(
                len(feed_dict_list), batch_size))
            stop_flag.value = 1
            continue

        if len(feed_dict_list) == batch_size:
            try:
                fetch_map_list = client.predict(
                    feed=feed_dict_list, fetch=["crf_decode"])
                for i, fetch_map in enumerate(fetch_map_list):
                    targets = fetch_map["crf_decode"][:unpad_seq_len]
                    outputqueue.put([
                        feed_dict_list[i]["src_ids"][1:unpad_seq_len - 1],
                        targets[1:-1]
                    ])
                line_list = []
                feed_dict_list = []
                transaction_failed_count = 0
            except Exception as e:
                transaction_failed_count += 1
                sys.stderr.write("Failed {} times with service {}\n".format(
                    transaction_failed_count, endpoint))
                sys.stderr.write(str(e) + '\n')
                time.sleep(min(1.0, 0.1 * transaction_failed_count))
                if transaction_failed_count > 20:
                    stop_flag.value = 1

    outputqueue.put(None)


class SeqLabelingDataset(dg.MultiSlotDataGenerator):
    def setup(self,
              host='127.0.0.1',
              port=9379,
              service_name='DistillService',
              require_num=1):
        self.bert_reader = BertSequenceReader(
            vocab_file="vocab.txt", max_seq_len=128)

        # {ip_port:[process, stop_flag], ... }
        self.endpoint2process = {}

        # connect balance server 127.0.0.1:9379
        self.client = BClient(host, port, service_name)
        nodes = self.client.start(require_num=require_num)
        sys.stderr.write('init servers={}\n'.format(nodes))

        queue_size = 5120
        self.input_queue = mps.Queue(queue_size)
        self.result_queue = mps.Queue(queue_size)
        for ip_port in nodes:
            sys.stderr.write('start:' + str(ip_port) + '\n')
            assert ip_port not in self.endpoint2process.keys()

            stop_flag = mps.Value('i', 0)
            p = mps.Process(
                target=client_producer,
                args=(
                    self.input_queue,
                    self.result_queue,
                    ip_port,
                    self.bert_reader,
                    stop_flag, ))
            self.endpoint2process[ip_port] = [p, stop_flag]

        for p, stop_flag in self.endpoint2process.values():
            p.daemon = True
            p.start()

        self._update_service_start()

    def _update_service_start(self):
        self._stop_update_service = False
        # Todo. client add callback?
        self._update_thread = Thread(target=self._update_service)
        self._update_thread.daemon = True
        self._update_thread.start()

    def _update_service(self):
        while not self._stop_update_service:
            time.sleep(3)
            is_update, nodes = self.client.get_teacher_list()
            if not is_update:
                continue

            old_nodes = self.endpoint2process.keys()

            # node need remove
            rm_nodes = set(old_nodes) - set(nodes)
            # node need add
            add_nodes = set(nodes) - set(old_nodes)

            # stop process
            for node in rm_nodes:
                if node not in self.endpoint2process.keys():
                    continue
                p, stop_flag = self.endpoint2process[node]
                stop_flag.value = 1
                p.join(timeout=1.2)

                # if p.is_alive():
                #     sys.stderr.write('node={} is alive'.format(node))
                # else:
                del self.endpoint2process[node]
                sys.stderr.write("remove service " + str(node) + '\n')

            # add process
            for node in add_nodes:
                if node in self.endpoint2process.keys():
                    continue
                stop_flag = mps.Value('i', 0)
                p = mps.Process(
                    target=client_producer,
                    args=(
                        self.input_queue,
                        self.result_queue,
                        node,
                        self.bert_reader,
                        stop_flag, ))
                self.endpoint2process[node] = [p, stop_flag]
                p.daemon = True
                p.start()
                sys.stderr.write("Add service " + str(node) + '\n')

    def run_from_stdin(self):
        dispatcher_thread = Thread(
            target=dispatcher,
            args=(
                self.input_queue,
                #sys.stdin,
                None,
                self.endpoint2process))
        dispatcher_thread.deamon = True
        dispatcher_thread.start()

        i = 0
        t1 = time.time()
        while True:
            if i == 400:
                period = time.time() - t1
                speed = 400 / period
                sys.stderr.write('speed={} line/s\n'.format(speed))
                t1 = time.time()
                i = 0

            i += 1

            result = self.result_queue.get()
            # Todo. end?
            if result == None:
                sys.stderr.write('result=None, end')
                break
            sample = ("words", result[0]), ("targets", result[1])
            #sys.stdout.write('.')
            #sys.stdout.write(self._gen_str(sample))

        self._stop_update_service = True
        self.client.stop()
        self._update_thread.join()

        dispatcher_thread.join()
        for p, stop_flag in self.endpoint2process.values():
            stop_flag.value = 1
            p.join()


if __name__ == "__main__":
    require_num = 1
    # print(str(sys.argv))
    if len(sys.argv) == 2:
        require_num = sys.argv[1]
    seq_label_dataset = SeqLabelingDataset()
    seq_label_dataset.setup(require_num=require_num)
    seq_label_dataset.run_from_stdin()
