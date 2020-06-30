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

import logging
import numpy as np
import os
import signal
import six
import sys
import time

from paddle_serving_client import Client
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
    PENDING = 'pending'
    ERROR = 'error'
    FINISHED = 'finished'

    def __init__(self, server_id, server, stop_event_id, state=PENDING):
        self.server_id = server_id
        self.server = server
        self.stop_event_id = stop_event_id
        self.state = state


def predict_manage_worker(process, server_queue, server_result_queue,
                          require_num, predict_stop_events, get_servers_fun,
                          stop_event, predict_cond):
    """ thread that manage predict worker """
    num_shutdown_process = [0]

    def shutdown_one_process():
        server_queue.put(None)
        num_shutdown_process[0] += 1

    server_id = 0  # not yet used
    server_to_item = dict()  # server to server_item
    idle_predict_num = require_num
    event_set = set()
    for i in range(require_num):
        event_set.add(i)

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
            stop_event_id = server_item.stop_event_id
            # set stop event
            if not predict_stop_events[stop_event_id].is_set():
                predict_stop_events[stop_event_id].set()
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
            event_id = event_set.pop()
            server_item = ServerItem(server_id, server, event_id)
            server_queue.put(server_item)
            server_to_item[server] = server_item
            server_id += 1
            logger.info('Adding server={}'.format(server))

        try:
            # server job stop, return back stop_event_id
            server_result_item = server_result_queue.get(timeout=2)
            stop_event_id = server_result_item.stop_event_id
            event_set.add(stop_event_id)
            del server_to_item[server_result_item.server]

            # clear event
            predict_stop_events[stop_event_id].clear()
            # directly use count
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

    with predict_cond:
        for predict_stop_event in predict_stop_events:
            predict_stop_event.set()
        predict_cond.notify_all()

    for i in range(require_num):
        shutdown_one_process()
        clean_queue(server_result_queue)

    for i in range(20):
        shutdown_process = 0
        for p in process:
            if not p.is_alive():
                shutdown_process += 1
        if shutdown_process == len(process):
            break
        time.sleep(1)


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


class PaddlePredictServer(PredictServer):
    def __init__(self, server, config_file, feeds, fetchs, max_failed_times=3):
        self._server = server
        self._config_file = config_file
        self._predict_feed_idxs = []
        self._predict_feed_shapes = dict()
        self._predict_feed_size = dict()
        self._feeds = feeds
        self._fetchs = fetchs
        self._max_failed_times = max_failed_times
        self.client = None
        self._has_predict = False
        logger.info((server, config_file, feeds, fetchs, max_failed_times))

        self._time_line = _TimeLine()

    def connect(self):
        """ connect success, return True, else return False"""
        try:
            client = Client()
            client.load_client_config(self._config_file)
            client.connect([self._server])
            self.client = client
        except Exception as e:
            logger.error('Exception when connect server={}, Exception is:'.
                         format(str(self._server)))
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
        self._time_line.reset()
        feed_map_list = self._preprocess(feed_data)
        self._time_line.record('predict_preprocess')

        fetch_map_list = None
        for i in range(self._max_failed_times):
            try:
                fetch_map_list = self.client.predict(
                    feed=feed_map_list, fetch=self._fetchs)
                if fetch_map_list is None:
                    raise Exception('fetch_map_list should not be None')
                break
            except Exception as e:
                logger.warning('Failed {} times with server={}'.format(
                    i + 1, self._server))
                logger.warning('Exception:\n{}'.format(str(e)))
                # time.sleep(0.1 * (i + 1))

        self._time_line.record('real_predict')

        if fetch_map_list is None:
            return False, None

        predict_data = self._postprocess(fetch_map_list, len(feed_data))
        self._time_line.record('postprocess')
        return True, predict_data

    def __del__(self):
        try:
            # avoid serving exit bug when hasn't predict
            if self.client is not None and self._has_predict:
                self.client.release()
        except Exception as e:
            logger.critical('Release client failed with server={}, '
                            'there may be an unknown error'.format(
                                self._server))
            logger.critical('Exception:\n{}'.format(str(e)))
        logger.warning('Stopped predict server={}'.format(self._server))


class _TestNopPaddlePredictServer(PaddlePredictServer):
    def connect(self):
        return True

    def predict(self, feed_data):
        predict_data = [tuple() for _ in range(len(feed_data))]
        return True, predict_data

    def __del__(self):
        pass


def predict_worker(server_queue, server_result_queue, working_predict_count,
                   in_queue, out_queue, feeds, fetchs, conf_file, stop_events,
                   predict_lock, global_finished_task, predict_cond):
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
        while True:
            # get server
            server_item = server_queue.get()
            if server_item is None:
                server_result_queue.put(None)
                return

            # predict
            success = predict_loop(server_item, working_predict_count,
                                   in_queue, out_queue, feeds, fetchs,
                                   conf_file, stop_events, predict_lock,
                                   global_finished_task, predict_cond)

            server_item.state = ServerItem.FINISHED if success else ServerItem.ERROR
            server_result_queue.put(server_item)
            logger.info('Stopped server={}'.format(server_item.server))
    except Exception as e:
        if signal_exit[0] is True:
            pass
        else:
            six.reraise(*sys.exc_info())


def predict_loop(server_item, working_predict_count, in_queue, out_queue,
                 feeds, fetchs, conf_file, stop_events, predict_lock,
                 global_finished_task, predict_cond):
    logger.info('connect server={}'.format(server_item.server))
    predict_server = PaddlePredictServer if _NOP_PREDICT_TEST is False else _TestNopPaddlePredictServer
    client = predict_server(server_item.server, conf_file, feeds, fetchs)
    if not client.connect():
        return False

    stop_event = stop_events[server_item.stop_event_id]
    with predict_lock:
        working_predict_count.value += 1

    time_line = _TimeLine()
    finished_task = 0
    # predict loop
    while not stop_event.is_set():
        data = in_queue.get()
        time_line.record('get_data')

        # Poison
        if isinstance(data, _PoisonPill):
            poison_pill = data
            all_worker_done = False

            with predict_lock:
                # accumulate success predict task count
                poison_pill.predict_count += finished_task
                poison_pill.predict_count += global_finished_task.value

                # clean local and global finished task
                finished_task = 0
                global_finished_task.value = 0

                # last process
                if working_predict_count.value == 1:
                    if poison_pill.predict_count == poison_pill.feed_count:
                        working_predict_count.value -= 1
                        logger.debug('pid={} write poison to complete queue'.
                                     format(os.getpid()))
                        all_worker_done = True
                    else:
                        # NOTE. some predict worker failed,
                        # there are still tasks that have not been processed.
                        assert poison_pill.predict_count < poison_pill.feed_count, \
                            "if failed, predict_count={} must < feed_count={}".\
                            format(poison_pill.predict_count, poison_pill.feed_count)

                        in_queue.put(poison_pill)  # write back poison pill
                        continue  # continue process failed task
                else:  # not last process
                    logger.debug('pid={} write poison back to ready'.format(
                        os.getpid()))
                    assert poison_pill.predict_count <= poison_pill.feed_count, \
                        "predict_count={} must <= feed_count={}".format(poison_pill.predict_count,
                                                                        poison_pill.feed_count)
                    working_predict_count.value -= 1

            with predict_cond:
                if all_worker_done is True:
                    out_queue.put(poison_pill)  # poison consumer
                else:
                    in_queue.put(poison_pill)  # poison other predict worker
                if stop_event.is_set():
                    break
                # wait next reader iter or last failed predict job
                predict_cond.wait()

            with predict_lock:
                # go on working
                working_predict_count.value += 1
            continue

        success, out_data = client_predict(client, data)
        time_line.record('predict')

        if not success:
            with predict_lock:
                global_finished_task.value += finished_task
                in_queue.put(data)  # write back failed task data
                # last process
                if working_predict_count.value == 1:
                    # NOTE. need notify other predict worker, or maybe deadlock
                    with predict_cond:
                        predict_cond.notify_all()
                working_predict_count.value -= 1
                return False

        out_queue.put(out_data)
        finished_task += 1
        time_line.record('put_data')

    # disconnect with server
    with predict_lock:
        global_finished_task.value += finished_task
        # last process
        if working_predict_count.value == 1:
            # NOTE. need notify other predict worker, or maybe deadlock.
            with predict_cond:
                predict_cond.notify_all()
        working_predict_count.value -= 1
    return True


def client_predict(client, data):
    # read_data format e.g. [(img, label, img1, label1), (img, label, img1, label1)]
    # predict_data format e.g. [(predict0, predict1), (predict0, predict1)]
    # out_data = read_data + predict_data, will be
    # [(img, label, img1, label1, predict0, predict1),
    #  (img, label, img1, label1, predict0, predict1)]
    task, read_data = data
    success, predict_data = client.predict(read_data)
    if not success:
        return False, None

    out_data = read_data
    for i in range(len(out_data)):
        out_data[i] += predict_data[i]
    return True, (task, out_data)


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
                slot_data += (np.asarray(read_data[j][i]), )
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
