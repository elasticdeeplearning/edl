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

import ctypes
import logging
import numpy as np
import os

from paddle_serving_client import Client
from .timeline import _TimeLine

# only for local test.
_NOP_PREDICT_TEST = False
POISON_PILL = -1

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class _ServerItem(object):
    PENDING = 'pending'
    ERROR = 'error'
    FINISHED = 'finished'

    def __init__(self, server_id, server, stop_event_id, state=PENDING):
        self.server_id = server_id
        self.server = server
        self.stop_event_id = stop_event_id
        self.state = state


class _PoisonPill:
    def __init__(self, feed_count, predict_count=0, complete_count=0):
        self.feed_count = feed_count
        self.predict_count = predict_count
        self.complete_count = complete_count


class TaskMeta(object):
    def __init__(self, task_id, batch_id, batch_end):
        self.task_id = task_id
        self.batch_id = batch_id
        self.batch_end = batch_end


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
        self._feeds = feeds
        self._fetchs = fetchs
        self._max_failed_times = max_failed_times
        self.client = None
        logging.info((server, config_file, feeds, fetchs, max_failed_times))

        self._time_line = _TimeLine()

    def connect(self):
        """ connect success, return True, else return False"""
        try:
            client = Client()
            client.load_client_config(self._config_file)
            client.connect([self._server])
            self.client = client
        except Exception as e:
            logging.error('Exception when connect server={}, Exception is:'.
                          format(str(self._server)))
            logging.error(str(e))
            return False

        self._predict_feed_idxs = []
        for feed_idx, feed_name in enumerate(self._feeds):
            if feed_name in self.client.get_feed_names():
                self._predict_feed_idxs.append(feed_idx)
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
                feed_map[self._feeds[feed_idx]] = feed_data[batch_idx][
                    feed_idx]
            feed_map_list.append(feed_map)

        logging.debug('predict feed_map_list len={}'.format(
            len(feed_map_list)))
        return feed_map_list

    def _postprocess(self, fetch_map_list):
        """ fetch_map_list(map): format e.g. {'predict0': np[bsize, ..], 'predict1': np[bsize, ..]}
        return [(predict0, predict1), (predict0, predict1)]
        """
        predict_data = [tuple() for _ in range(len(self._fetchs))]
        for fetch_idx, fetch_name in enumerate(self._fetchs):
            batch_fetch_data = fetch_map_list[fetch_name]
            for fetch_data in batch_fetch_data:
                predict_data[fetch_idx] += (fetch_data, )
        return predict_data

    def predict(self, feed_data):
        """ predict success, return (True, predict_data),
        else return (False, None)"""
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
                logging.warning('Failed {} times with server={}'.format(
                    i + 1, self._server))
                logging.warning('Exception:\n{}'.format(str(e)))
                # time.sleep(0.1 * (i + 1))

        self._time_line.record('real_predict')

        if fetch_map_list is None:
            return False, None

        predict_data = self._postprocess(fetch_map_list)
        self._time_line.record('postprocess')
        return True, predict_data

    def __del__(self):
        try:
            if self.client is not None:
                self.client.release()
        except Exception as e:
            logging.critical('Release client failed with server={}, '
                             'there may be an unknown error'.format(
                                 self._server))
            logging.critical('Exception:\n{}'.format(str(e)))
        logging.warning('Stopped predict server={}'.format(self._server))


class _TestNopPaddlePredictServer(PaddlePredictServer):
    def connect(self):
        return True

    def predict(self, feed_data):
        predict_data = [tuple() for _ in range(len(self._fetchs))]
        return True, predict_data

    def __del__(self):
        pass


def predict_worker(server_queue, server_result_queue, working_predict_count,
                   in_queue, out_queue, feeds, fetchs, conf_file, stop_events,
                   predict_lock, global_finished_task, predict_cond):
    while True:
        # get server
        server_item = server_queue.get()
        if server_item is None:
            server_queue.put(None)  # poison_pill
            return

        # predict
        success = predict_loop(server_item, working_predict_count, in_queue,
                               out_queue, feeds, fetchs, conf_file,
                               stop_events, predict_lock, global_finished_task,
                               predict_cond)

        server_item.state = _ServerItem.FINISHED if success else _ServerItem.ERROR
        server_result_queue.put(server_item)
        logging.info('Stopped server={}'.format(server_item.server))


def predict_loop(server_item, working_predict_count, in_queue, out_queue,
                 feeds, fetchs, conf_file, stop_events, predict_lock,
                 global_finished_task, predict_cond):
    logging.info('connect server={}'.format(server_item.server))
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
                        logging.debug('pid={} write poison to complete queue'.
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
                    logging.debug('pid={} write poison back to ready'.format(
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
        time_line.recode('put_data')

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
    task_idx, read_data = data
    success, predict_data = client.predict(read_data)
    if not success:
        return False, None

    out_data = read_data
    for i in range(len(data)):
        out_data[i] += predict_data[i]
    return True, (task_idx, read_data)


class ReaderType(object):
    SAMPLE = 0
    SAMPLE_LIST = 1
    BATCH = 2


def reader_worker(reader, reader_type, out_queue, stop_event, task_semaphore,
                  reader_cond):
    while not stop_event.is_set():
        send_idx = 0
        # TODO. reader_type
        for read_data in reader():
            # Use task_semaphore to keep order.
            # e.g. If semaphore is 4, reader send task(0, 1, 2, 3),
            # consumer may recv out-of-order task(3, 1, 2) before task(0), consumer will store then,
            # when task(0) is completed and consumer recv it, it will release semaphore,
            # reader go on working.
            task_semaphore.acquire()
            out_queue.put((send_idx, read_data))
            send_idx += 1

        poison_pill = _PoisonPill(send_idx)

        with reader_cond:
            out_queue.put(poison_pill)
            # wait next reader iter
            reader_cond.wait()


def fetch_worker(reader_type, in_queue, out_queue, stop_event, task_semaphore,
                 fetch_cond):
    logging.info('fetch_worker pid={}'.format(os.getpid()))
    task_data = dict()

    while not stop_event.is_set():
        recv_idx = 0

        recv_data = in_queue.get()
        if isinstance(recv_data, _PoisonPill):
            poison_pill = recv_data
            assert poison_pill.feed_count == poison_pill.predict_count, \
                "poison_pill feed_count={} != predict_count={}".\
                format(poison_pill.feed_count, poison_pill.predict_count)

            if recv_idx == poison_pill.predict_count:
                # all fetch job success
                poison_pill.complete_count = recv_idx
                out_queue.put(poison_pill)
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
                logging.debug('fetch is unordered!!!')
                in_queue.put(poison_pill)  # write back poison

        task_id, data = recv_data
        if task_id == recv_idx:
            task_semaphore.release()
            recv_idx += 1
            # TODO. joint
        else:
            # store out-of-order samples
            task_data[task_id] = data


def fetch_worker(batch_size, d_batch_size, shared_memory_array, idle_queue,
                 complete_queue, out_idle_queue, out_ready_queue, fetch_cond):
    logging.info('fetch_worker pid={}'.format(os.getpid()))
    # init
    out_idx = out_idle_queue.get()
    assert out_idx == 0, "first out_idx must == 0"
    out_offset = 0
    last_complete_memory_idx, last_length = -1, -1

    out_uid = 0
    complete_count = 0
    while True:
        complete_memory_idx = complete_queue.get()

        # Poison
        if isinstance(complete_memory_idx, _PoisonPill):
            poison_pill = complete_memory_idx
            assert poison_pill.feed_count == poison_pill.predict_count, \
                "poison_pill feed_count={} != predict_count={}".format(poison_pill.feed_count,
                                                                       poison_pill.predict_count)
            poison_pill.complete_count += complete_count
            complete_count = 0  # clear

            # last no full batch
            if last_complete_memory_idx != -1 and poison_pill.complete_count + 1 == poison_pill.predict_count:
                shared_memory_array.copy_shm_to_out(last_complete_memory_idx,
                                                    out_idx, out_offset,
                                                    out_uid, last_length)
                idle_queue.put(last_complete_memory_idx)

                last_complete_memory_idx = -1
                out_offset += last_length

                poison_pill.complete_count += 1

            if poison_pill.complete_count == poison_pill.predict_count:  # all fetch job success
                assert last_complete_memory_idx == -1, "no full batch fetch failed?"
                if out_offset > 0:
                    # put last no full ready data to out_ready_queue
                    out_ready_queue.put(out_idx)
                else:
                    # write back idle out idx
                    out_idle_queue.put(out_idx)

                # poison distill reader
                # self._out_ready_queue.put(DistillReader.POISON_PILL)  # NOTE. POISON here may be hang
                with fetch_cond:
                    # NOTE!!! put poison pill can only be placed inside the critical area of the cond variable,
                    # otherwise, when reader finishes, reentrant and sending notify,
                    # the current process may not have entered the critical area and cannot receive notifications.
                    # The will hang.
                    out_ready_queue.put(POISON_PILL)
                    fetch_cond.wait()

                # init
                out_idx = out_idle_queue.get()
                out_offset = 0
                last_complete_memory_idx, last_length = -1, -1
                out_uid = 0
                continue
            else:
                # NOTE!!!!!!!!!
                # When predict with multi process worker, complete_queue maybe unordered.
                # Last process may write POISON_PILL before other process.
                # For example:
                # [DEBUG 2020-04-20 23:37:37,582 distill_reader.py:649] pid=60636 write ready_memory_idx=15 memory_uid=[47L, 4L]
                # [DEBUG 2020-04-20 23:37:37,582 distill_reader.py:649] pid=60637 write ready_memory_idx=0 memory_uid=[48L, 2L]
                # [DEBUG 2020-04-20 23:37:37,582 distill_reader.py:603] pid=60636 write poison back to ready
                # [DEBUG 2020-04-20 23:37:37,583 distill_reader.py:599] pid=60637 write poison to complete queue
                # From time order, we must want complete_queue be Queue(15, 0, poison_pill).
                # But in fact complete_queue may be Queue(0, poison_pill, 15), for the queue.put() is unordered in multi process.
                assert poison_pill.complete_count < poison_pill.predict_count, \
                    "poison_pill complete_count={} must< predict_count={}".format(poison_pill.complete_count,
                                                                                  poison_pill.predict_count)
                logging.debug('complete is unordered!!!!')
                complete_queue.put(poison_pill)  # write back poison
                continue

        uid_list = shared_memory_array.get_shared_uid(complete_memory_idx)
        uid, filled_length = uid_list
        logging.debug('result_memory_idx={} memory_uid={}'.format(
            complete_memory_idx, uid_list))

        assert filled_length <= d_batch_size
        # not full, must be last batch
        if filled_length != d_batch_size:
            logging.debug('filled_length={} d_batch_size={} idx={}'.format(
                filled_length, d_batch_size, complete_memory_idx))
            last_complete_memory_idx = complete_memory_idx
            last_length = filled_length
        else:  # full
            shared_memory_array.copy_shm_to_out(complete_memory_idx, out_idx,
                                                out_offset, out_uid,
                                                d_batch_size)
            # copy done, write back idx to idle queue
            idle_queue.put(complete_memory_idx)
            complete_count += 1

            out_offset += d_batch_size
            # out is ready
            if out_offset == batch_size:
                out_ready_queue.put(out_idx)
                out_idx = out_idle_queue.get()
                out_uid += 1
                out_offset = 0
