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

import gc
import logging
import numpy as np
import os
import signal
import six
import sys
import time
import threading

from paddle_serving_client import MultiLangClient
# from paddle_serving_client import Client
from six.moves import queue
from six.moves import reduce
from .timeline import _TimeLine
from ..discovery.server_alive import is_server_alive

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

# only for local test.
_NOP_PREDICT_TEST = False


def _is_server_alive(server):
    # only for test, need find a better test method
    if _NOP_PREDICT_TEST:
        return True
    return is_server_alive(server)[0]


class ServerItem(object):
    RUNNING = 'pending'
    ERROR = 'error'
    FINISHED = 'finished'
    STOPPING = 'stopping'

    def __init__(self, server_id, server, state=RUNNING):
        self.server_id = server_id
        self.server = server
        self.state = state


def predict_manage_worker(server_queue, server_result_queue, require_num,
                          get_servers_fun, stop_event):
    """ thread that manage predict worker """
    server_id = 0  # not yet used
    server_to_item = dict()  # server to server_item
    idle_predict_num = require_num

    # Fix the order of object destruction
    first_in = True
    while not stop_event.is_set():
        servers = get_servers_fun(first_in)
        if servers is None:
            break
        first_in = False

        servers = set(servers)
        old_servers = set(server_to_item.keys())

        add_servers = servers - old_servers
        rm_servers = old_servers - servers

        # Remove servers
        while len(rm_servers) != 0:
            server = rm_servers.pop()
            server_item = server_to_item[server]

            # need stop
            if server_item.state != ServerItem.STOPPING:
                server_item.state = ServerItem.STOPPING
                server_queue.put(server_item)
                logger.info('Removing server={}'.format(server))

        # Add servers
        while len(add_servers) != 0:
            if idle_predict_num == 0:
                break
            assert idle_predict_num > 0, 'idle_predict_num must > 0'

            server = add_servers.pop()
            if not _is_server_alive(server):
                logger.warning('server={} is not alive'.format(server))
                continue

            idle_predict_num -= 1
            server_item = ServerItem(server_id, server)
            server_queue.put(server_item)
            server_to_item[server] = server_item
            server_id += 1
            logger.info('Adding server={}'.format(server))

        try:
            # server job stop, return back stop_event_id
            server_result_item = server_result_queue.get(timeout=2)
            del server_to_item[server_result_item.server]

            idle_predict_num += 1
            assert idle_predict_num <= require_num, \
                'idle_predict_num={} must <= require_num={}'.format(
                    idle_predict_num, require_num)
            logger.info('Removed server={}'.format(server_result_item.server))
        except queue.Empty:
            pass

    def clean_queue(data_queue):
        while True:
            try:
                data_queue.get_nowait()
            except Exception:
                break

    clean_queue(server_queue)
    clean_queue(server_result_queue)


class _PoisonPill:
    def __init__(self, feed_count, predict_count=0):
        self.feed_count = feed_count
        self.predict_count = predict_count


class Task(object):
    def __init__(self, task_id, batch_id=-1, batch_size=-1):
        self.task_id = task_id
        self.batch_id = batch_id
        self.batch_size = batch_size


class PredictServer(object):
    def connect(self):
        """ connect success, return True, else return False"""
        raise NotImplementedError()

    def predict(self, feed_data):
        """ predict success, return True, else return False"""
        raise NotImplementedError()


class AsyncPredictClient(object):
    def __init__(self, server, config_file, feeds, fetchs):
        self.server = server
        self._config_file = config_file
        self._predict_feed_idxs = []
        self._predict_feed_shapes = dict()
        self._predict_feed_size = dict()
        self._feeds = feeds
        self._fetchs = fetchs
        self.client = None
        self._has_predict = False
        self.need_stop = False

        logger.info((server, config_file, feeds, fetchs))

    def connect(self):
        """ connect success, return True, else return False"""
        try:
            client = MultiLangClient()
            #client.load_client_config(self._config_file)
            client.connect([self.server])
            self.client = client
        except Exception as e:
            logger.error('Exception when connect server={}, Exception is:'.
                         format(str(self.server)))
            logger.error(str(e))
            return False

        self._predict_feed_idxs = []
        self._predict_feed_shapes = dict()
        self._predict_feed_size = dict()
        for feed_idx, feed_name in enumerate(self._feeds):
            if feed_name in self.client.get_feed_names():
                self._predict_feed_idxs.append(feed_idx)
                self._predict_feed_shapes[feed_name] = tuple(
                    self.client.feed_shapes_[feed_name])
                self._predict_feed_size[feed_name] = reduce(
                    lambda x, y: x * y, self._predict_feed_shapes[feed_name])
        return True

    def _preprocess(self, feed_data):
        """ feed_data(list). format e.g. [(img, label, img1, label1), (img, label, img1, label1)]
        However, predict may only need (img, img1).
        return [{'img': img, 'img1': img1}, {'img': img, 'img1': img1}]
        """
        feed_map_list = []
        for batch_idx in range(len(feed_data)):
            feed_map = dict()
            for feed_idx in self._predict_feed_idxs:
                feed_name = self._feeds[feed_idx]
                feed_size = self._predict_feed_size[feed_name]
                feed_shape = self._predict_feed_shapes[feed_name]

                data = feed_data[batch_idx][feed_idx]
                if data.size == feed_size:
                    data = data.reshape(feed_shape)

                feed_map[feed_name] = data
            feed_map_list.append(feed_map)

        logger.debug('predict feed_map_list len={}'.format(len(feed_map_list)))
        return feed_map_list

    def _postprocess(self, fetch_map_list, batch_size):
        """ fetch_map_list(map): format e.g. {'predict0': np[bsize, ..], 'predict1': np[bsize, ..]}
        return [(predict0, predict1), (predict0, predict1)]
        """
        predict_data = [tuple() for _ in range(batch_size)]
        for fetch_name in self._fetchs:
            batch_fetch_data = fetch_map_list[fetch_name]
            for batch_idx, fetch_data in enumerate(batch_fetch_data):
                predict_data[batch_idx] += (fetch_data, )
        return predict_data

    def predict(self, feed_data):
        """ predict success, return (True, predict_data),
        else return (False, None)"""
        self._has_predict = True
        feed_map_list = self._preprocess(feed_data)
        future = self.client.predict(
            feed=feed_map_list, fetch=self._fetchs, asyn=True)
        return future

    def result(self, future, feed_data):
        fetch_map_list = None
        try:
            fetch_map_list = future.result()
        except Exception as e:
            logger.warning('Failed with server={}'.format(self.server))
            logger.warning('Exception:\n{}'.format(str(e)))

        if fetch_map_list is None:
            return False, None

        if fetch_map_list['status_code'] != 0:
            logger.warning('Failed status code={}'.format(fetch_map_list[
                'status_code']))
            return False, None

        predict_data = self._postprocess(fetch_map_list, len(feed_data))
        return True, predict_data

    def __del__(self):
        try:
            # avoid serving exit bug when hasn't predict
            if self.client is not None and self._has_predict:
                self.client.release()
        except Exception as e:
            logger.critical('Release client failed with server={}, '
                            'there may be an unknown error'.format(
                                self.server))
            logger.critical('Exception:\n{}'.format(str(e)))
        logger.warning('Stopped predict server={}'.format(self.server))


class _TestNopAsyncPredictClient(AsyncPredictClient):
    class _Future(object):
        def __init__(self, feed_data):
            self._feed_data = feed_data

        def add_done_callback(self, call_back):
            call_back(self)

    def connect(self):
        return True

    def predict(self, feed_data):
        return self._Future(feed_data)

    def result(self, future, feed_data):
        predict_data = [tuple() for _ in range(len(feed_data))]
        return True, predict_data


class PredictPool(object):
    def __init__(self):
        self._client_queue = queue.Queue()
        self._server_to_clients = dict()

    def add_client(self, server_item, feeds, fetchs, conf_file, concurrent=3):
        server = server_item.server
        if server_item.server in self._server_to_clients:
            logger.warning('server={} in predict client?'.format(
                server_item.server))
            return True

        predict_server = AsyncPredictClient if _NOP_PREDICT_TEST is False else _TestNopAsyncPredictClient
        client = predict_server(server, conf_file, feeds, fetchs)
        if not client.connect():
            return False

        self._server_to_clients[server] = (server_item, client)
        for _ in range(concurrent):
            self._client_queue.put(client)
        return True

    def stop_client(self, server_item):
        server = server_item.server
        item_client = self._server_to_clients.get(server)
        # client may removed before this
        if item_client is not None:
            _, client = item_client
            client.need_stop = True

    def rm_client(self, client, server_result_queue):
        server = client.server
        item_client = self._server_to_clients.get(server)
        # client already removed
        if item_client is None:
            return
        server_item, client = item_client
        del self._server_to_clients[server]

        server_item.state = ServerItem.FINISHED
        server_result_queue.put(server_item)

    def run(self, in_queue, out_queue, server_result_queue):
        finished_task_count_lock = threading.Lock()
        finished_task_count = [0]

        while True:
            data = in_queue.get()
            if isinstance(data, _PoisonPill):
                poison_pill = data
                if finished_task_count[0] == poison_pill.feed_count:
                    poison_pill.predict_count = poison_pill.feed_count
                    return poison_pill  # all task finished

                in_queue.put(data)  # write back poison pill
                time.sleep(0.003)  # wait 3ms
                continue  # continue process failed task

            task, read_data = data
            while True:
                client = self._client_queue.get()
                if client.need_stop:
                    self.rm_client(client, server_result_queue)
                    continue

                # FIXME. may failed
                future = client.predict(read_data)

                call_back = predict_call_back(in_queue, out_queue, client,
                                              data, finished_task_count_lock,
                                              finished_task_count,
                                              self._client_queue)
                future.add_done_callback(call_back)
                #logger.info('garbage collector output is {}'.format(gc.get_stats()))
                #gc.collect(0)
                break


def predict_call_back(
        in_queue,
        out_queue,
        client,
        data,
        finished_task_count_lock,
        finished_task_count,
        client_queue, ):
    def _call_back(future):
        task, read_data = data
        batch_size = len(read_data)
        success = False
        try:
            success, predict_data = client.result(future, read_data)
        except Exception as e:
            logger.info('predict error={}'.format(e))
        if not success:
            in_queue.put(data)
            client.need_stop = True

        client_queue.put(client)  # complete, put back
        if not success:
            return

        out_data = read_data
        for i in range(batch_size):
            out_data[i] += predict_data[i]

        out_queue.put((task, out_data))
        with finished_task_count_lock:
            finished_task_count[0] += 1

    return _call_back


def predict_process(server_queue, server_result_queue, in_queue, out_queue,
                    feeds, fetchs, conf_file, stop_event, predict_cond):
    logger.info('predict process pid={}'.format(os.getpid()))
    signal_exit = [False, ]

    # Define signal handler function
    def predict_signal_handle(signum, frame):
        signal_exit[0] = True
        # fix infinite reader hang
        server_queue.cancel_join_thread()
        server_result_queue.cancel_join_thread()
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        exit(0)

    # register signal.SIGTERM's handler
    signal.signal(signal.SIGTERM, predict_signal_handle)

    try:
        client_pool = PredictPool()
        manager_need_stop = threading.Event()

        def server_manager():
            while not manager_need_stop.is_set():
                try:
                    server_item = server_queue.get(timeout=2)
                except queue.Empty:
                    continue

                if server_item is None:
                    server_result_queue.put(None)
                    return

                if server_item.state == ServerItem.RUNNING:
                    if not client_pool.add_client(server_item, feeds, fetchs,
                                                  conf_file):
                        server_item.state = ServerItem.FINISHED
                        server_result_queue.put(server_item)
                elif server_item.state == ServerItem.STOPPING:
                    client_pool.stop_client(server_item)

        manage_thread = threading.Thread(target=server_manager, )
        manage_thread.daemon = True
        manage_thread.start()

        #gc.set_debug(gc.DEBUG_LEAK)

        while not stop_event.is_set():
            poison_pill = client_pool.run(in_queue, out_queue,
                                          server_result_queue)
            with predict_cond:
                out_queue.put(poison_pill)
                predict_cond.wait()

        manager_need_stop.set()
        manage_thread.join()
    except Exception as e:
        if signal_exit[0] is True:
            pass
        else:
            print('error={}'.format(e))
            six.reraise(*sys.exc_info())


class ReaderType(object):
    SAMPLE = 0
    SAMPLE_LIST = 1
    BATCH = 2


def reader_worker(reader, reader_type, teacher_batch_size, out_queue,
                  stop_event, task_semaphore, reader_cond):
    # Use task_semaphore to keep order.
    # e.g. If semaphore is 4, reader send task(0, 1, 2, 3),
    # consumer may recv out-of-order task(3, 1, 2) before task(0), consumer will store then,
    # when task(0) is completed and consumer recv it, it will release semaphore,
    # reader go on working.

    signal_exit = [False, ]

    def reader_signal_handle(signum, frame):
        signal_exit[0] = True
        out_queue.cancel_join_thread()
        exit(0)

    # register signal.SIGTERM's handler
    signal.signal(signal.SIGTERM, reader_signal_handle)

    read_func_map = {
        ReaderType.SAMPLE: read_sample,
        ReaderType.SAMPLE_LIST: read_sample_list,
        ReaderType.BATCH: read_batch
    }
    read_func = read_func_map[reader_type]

    try:
        while not stop_event.is_set():
            task_size = read_func(reader, teacher_batch_size, out_queue,
                                  task_semaphore)

            poison_pill = _PoisonPill(task_size)
            with reader_cond:
                out_queue.put(poison_pill)
                if stop_event.is_set():
                    break
                # wait next reader iter
                reader_cond.wait()
    except Exception as e:
        if signal_exit[0] is True:
            pass
        else:
            six.reraise(*sys.exc_info())


def read_sample(reader, teacher_batch_size, out_queue, task_semaphore):
    task_id = 0
    sample_size = 0
    send_data = []

    for read_data in reader():
        # read_data: (img, label)
        slot_data = tuple()
        for slot in read_data:
            slot_data += (np.asarray(slot), )
        send_data.append(slot_data)

        sample_size += 1
        if sample_size == teacher_batch_size:
            task_semaphore.acquire()
            task = Task(task_id=task_id)
            out_queue.put((task, send_data))

            task_id += 1
            send_data = []
            sample_size = 0
        else:
            continue

    # remain
    if sample_size != 0:
        task_semaphore.acquire()
        task = Task(task_id=task_id)
        out_queue.put((task, send_data))
        task_id += 1

    return task_id


def read_sample_list(reader, teacher_batch_size, out_queue, task_semaphore):
    task_id = 0
    batch_id = 0
    sample_size = 0
    send_data = []

    for read_data in reader():
        # read_data: [(img, label), (img, label), .. ]
        batch_size = len(read_data)
        for sample_data in read_data:
            # sample_data: (img, label)
            slot_data = tuple()
            for slot in sample_data:
                slot_data += (np.asarray(slot), )
            send_data.append(slot_data)

            sample_size += 1
            if sample_size == teacher_batch_size:
                task_semaphore.acquire()
                task = Task(
                    task_id=task_id, batch_id=batch_id, batch_size=batch_size)
                out_queue.put((task, send_data))

                task_id += 1
                send_data = []
                sample_size = 0
            else:
                continue

        # remain
        if sample_size != 0:
            task_semaphore.acquire()
            task = Task(
                task_id=task_id, batch_id=batch_id, batch_size=batch_size)
            out_queue.put((task, send_data))

            task_id += 1
            send_data = []
            sample_size = 0

        batch_id += 1

    return task_id


def read_batch(reader, teacher_batch_size, out_queue, task_semaphore):
    task_id = 0
    batch_id = 0
    sample_size = 0
    send_data = []

    for read_data in reader():
        # read_data: (img[batch, shape], label[batch, shape])
        slot_size = len(read_data)
        batch_size = len(read_data[0])

        for i in range(batch_size):
            slot_data = tuple()
            for j in range(slot_size):
                slot_data += (read_data[j][i], )
            send_data.append(slot_data)

            sample_size += 1
            if sample_size == teacher_batch_size:
                task_semaphore.acquire()
                task = Task(
                    task_id=task_id, batch_id=batch_id, batch_size=batch_size)
                out_queue.put((task, send_data))

                task_id += 1
                send_data = []
                sample_size = 0
            else:
                continue

        # remain
        if sample_size != 0:
            task_semaphore.acquire()
            task = Task(
                task_id=task_id, batch_id=batch_id, batch_size=batch_size)
            out_queue.put((task, send_data))

            task_id += 1
            send_data = []
            sample_size = 0

        batch_size += 1

    return task_id


def fetch_out(reader_type, in_queue, stop_event, task_semaphore):
    fetch_func_map = {
        ReaderType.SAMPLE: fetch_sample,
        ReaderType.SAMPLE_LIST: fetch_sample_list,
        ReaderType.BATCH: fetch_batch
    }
    fetch_func = fetch_func_map[reader_type]

    class RecvId:
        def __init__(self, val):
            self.val = val

    class Samples:
        def __init__(self, size, sample_list):
            self.size = size
            self.samples = sample_list

    store_data = dict()
    recv_id = RecvId(0)
    samples = Samples(0, [])  # (accumulate_sample_size, samples)
    while not stop_event.is_set():
        fetch_data = in_queue.get()
        if isinstance(fetch_data, _PoisonPill):
            poison_pill = fetch_data
            assert poison_pill.feed_count == poison_pill.predict_count, \
                "poison_pill feed_count={} != predict_count={}". \
                format(poison_pill.feed_count, poison_pill.predict_count)

            if recv_id.val == poison_pill.predict_count:
                return
            else:
                # NOTE! When predict with multi process, predict_out_queue maybe unordered.
                # Last process may write POISON_PILL before other process. For example:
                # time=0, pid=0 predict_out_queue.put([img0])
                # time=1, pid=1 predict_out_queue.put([img1])
                # time3, pid=0 predict_in_queue.put(poison_pill)
                # time4, pid=1 predict_out_queue.put(poison_pill)
                # From time order, we must want predict_out_queue be ([img0], [img1], poison_pill)
                # But in fact predict_out_queue may be Queue([img1], poison_pill, [img0]),
                # for the queue.put() is unordered in multi process.
                logger.debug('fetch is unordered!!!')
                in_queue.put(poison_pill)  # write back poison
                continue

        for data in fetch_func(fetch_data, store_data, recv_id, task_semaphore,
                               samples):
            yield data


def fetch_sample(fetch_data, store_data, recv_id, task_semaphore, samples):
    # data: (img, label, predict)
    task, data = fetch_data

    # store data, may out-of-order
    store_data[task.task_id] = data
    while True:
        if recv_id.val in store_data:
            task_semaphore.release()
            recv_data = store_data.pop(recv_id.val)
            recv_id.val += 1
            for sample_data in recv_data:
                yield sample_data
        else:
            break


def fetch_sample_list(fetch_data, store_data, recv_id, task_semaphore,
                      samples):
    # data: [(img, label, predict), (img, label, predict), ..]
    task, data = fetch_data

    # store data, may out-of-order
    store_data[task.task_id] = fetch_data
    while True:
        if recv_id.val in store_data:
            task_semaphore.release()
            recv_task, recv_data = store_data.pop(recv_id.val)
            recv_id.val += 1

            samples.size += len(recv_data)
            # joint batch list
            samples.samples += recv_data
            assert samples.size <= recv_task.batch_size
            if samples.size == recv_task.batch_size:
                # out_data: batch sample [(img, label, predict), (img, label, predict), ..,]
                yield samples.samples
                samples.size = 0
                samples.samples = []
        else:
            break


def fetch_batch(fetch_data, store_data, recv_id, task_semaphore, samples):
    # data: [(img, label, predict), (img, label, predict), ..]
    task, data = fetch_data

    # store data, may out-of-order
    store_data[task.task_id] = fetch_data
    while True:
        if recv_id.val in store_data:
            task_semaphore.release()
            recv_task, recv_data = store_data.pop(recv_id.val)
            recv_id.val += 1

            samples.size += len(recv_data)
            # joint batch list
            samples.samples += recv_data
            assert samples.size <= recv_task.batch_size
            if samples.size == recv_task.batch_size:
                # len (img, label, predict)
                slot_size = len(samples.samples[0])
                batch_size = recv_task.batch_size

                batch_data = tuple()
                for i in range(slot_size):
                    slot = []
                    for j in range(batch_size):
                        slot.append(samples.samples[j][i])
                    batch_data += (np.array(slot), )

                # out_data: (img[batch, shape], label[batch, shape], sample[batch, shape])
                yield batch_data
                samples.size = 0
                samples.samples = []
        else:
            break
