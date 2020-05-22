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


class PredictServer(object):
    def connect(self):
        """ connect success, return True, else return False"""
        raise NotImplementedError()

    def predict(self, batch_size, feeds_memory, fetchs_memory):
        """ predict success, return True, else return False"""
        raise NotImplementedError()


class PaddlePredictServer(PredictServer):
    def __init__(self, server, config_file, feeds, fetchs, max_failed_times=3):
        self._server = server
        self._config_file = config_file
        self._feeds = feeds
        self._fetchs = fetchs
        self._fetchs_name = [fetch_var.name for fetch_var in fetchs]
        self._max_failed_times = max_failed_times
        self.client = None
        logging.info((server, config_file, feeds, fetchs, max_failed_times))

        self._time_line = _TimeLine()

    def connect(self):
        """ connect success, return True, else return False"""
        try:
            self.client = Client()
            self.client.load_client_config(self._config_file)
            self.client.connect([self._server])
        except Exception as e:
            logging.error('Exception when connect server={}, Exception is:'.
                          format(str(self._server)))
            logging.error(str(e))
            return False
        return True

    def _preprocess(self, batch_size, memorys):
        assert len(self._feeds) == len(memorys), \
            "Error: len(feeds_name)={} != len(memorys)={}".format(len(self._feeds), len(memorys))

        feed_map_list = []
        for batch_idx in range(batch_size):
            feed_map = {}
            for feed_idx, feed_var in enumerate(self._feeds):
                # NOTE. this method fast 5 times than list method.
                # with bs=2, shape=3*224*224, list time=46ms, slice time=9ms, np tolist 2-3ms
                # feed_map[name] = list(memorys[feed_idx][batch_idx])
                # feed_map[var.name] = memorys[feed_idx][batch_idx][:]
                feed_map[feed_var.name] = np.frombuffer(
                    memorys[feed_idx][batch_idx],
                    dtype=feed_var.dtype).reshape(feed_var.shape)
            feed_map_list.append(feed_map)
        logging.debug('predict feed_map_list len={}'.format(
            len(feed_map_list)))
        return feed_map_list

    def _postprocess(self, batch_size, memorys, fetchs_map):
        assert len(self._fetchs) == len(fetchs_map), \
            "Error: len(fetchs)={} != len(fetchs_list)={}".format(len(self._fetchs), len(fetchs_map))

        for fetch_idx, fetch_var in enumerate(self._fetchs):
            # if fetch is list, use slice
            # >>> a_type = ctypes.c_float * 4
            # >>> a = a_type()
            # >>> a[:]
            # [0.0, 0.0, 0.0, 0.0]
            # >>> a[:] = [1.0, 3.0, 2.0, 5.0]
            # >>> a[:]
            # [1.0, 3.0, 2.0, 5.0]
            fetch_data = fetchs_map[fetch_var.name]
            assert ctypes.sizeof(memorys[fetch_idx][0]) * batch_size == fetch_data.nbytes, \
                "sizeof(memory)={} != fetch_data.nbytes={} shape={} dtype={}".format(
                    ctypes.sizeof(memorys[fetch_idx][0]) * batch_size,
                    fetch_data.nbytes,
                    fetch_data.shape,
                    fetch_data.dtype)
            ctypes.memmove(memorys[fetch_idx], fetch_data.ctypes.data,
                           fetch_data.nbytes)

    def predict(self, batch_size, feeds_memory, fetchs_memory):
        """ predict success, return True, else return False"""
        self._time_line.reset()
        feed_map_list = self._preprocess(batch_size, feeds_memory)
        self._time_line.record('preprocess')

        fetch_map_list = None
        for i in range(self._max_failed_times):
            try:
                fetch_map_list = self.client.predict(
                    feed=feed_map_list, fetch=self._fetchs_name)
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
            return False

        self._postprocess(batch_size, fetchs_memory, fetch_map_list)
        self._time_line.record('postprocess')
        return True

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

    def predict(self, batch_size, feeds_memory, fetchs_memory):
        feed_map_list = self._preprocess(batch_size, feeds_memory)
        return True

    def __del__(self):
        pass


def feed_worker(reader, idle_queue, ready_queue, shared_memory_array,
                feed_cond):
    """ supported samples format [np_var0, np_var1, ..] each numpy contains <=d_batch_size """
    # TODO. exit feed_worker
    while True:
        uid = 0
        for d_batch in reader():
            idle_memory_idx = idle_queue.get()
            shared_memory_array.write_feed(idle_memory_idx, uid, d_batch)
            ready_queue.put(idle_memory_idx)
            uid += 1

        poison_pill = _PoisonPill(uid)

        # self._ready_queue.put(poison_pill)  # NOTE! can't put here, or will hang
        with feed_cond:
            # NOTE! see comment in _fetch_cond.wait()
            ready_queue.put(poison_pill)
            feed_cond.wait()


def predict_worker(predict_server_queue, predict_server_result_queue,
                   shared_memory_array, serving_conf_file, predict_stop_events,
                   predict_job_lock, count_of_working_predict, ready_queue,
                   predict_job_count, predict_cond, complete_queue):
    while True:
        # get server item
        server_item = predict_server_queue.get(block=True)
        if server_item is None:
            predict_server_result_queue.put(None)
            return

        # predict
        is_normal_stop = predict(
            server_item, shared_memory_array, serving_conf_file,
            predict_stop_events, predict_job_lock, count_of_working_predict,
            ready_queue, predict_job_count, predict_cond, complete_queue)

        server_item.state = _ServerItem.FINISHED \
            if is_normal_stop else _ServerItem.ERROR

        # clear event, return result, release semaphore
        predict_server_result_queue.put(server_item)
        logging.info('Stopped server={}'.format(server_item.server))


def predict(
        server_item,
        shared_memory_array,
        serving_conf_file,
        predict_stop_events,
        predict_job_lock,
        count_of_working_predict,
        ready_queue,
        predict_job_count,
        predict_cond,
        complete_queue, ):
    feeds = shared_memory_array.get_predict_feeds()
    fetchs = shared_memory_array.get_fetchs()

    logging.info('connect server={}'.format(server_item.server))
    predict_server = PaddlePredictServer if _NOP_PREDICT_TEST is False else _TestNopPaddlePredictServer
    client = predict_server(server_item.server, serving_conf_file, feeds,
                            fetchs)
    if client.connect() is False:
        return False

    stop_event = predict_stop_events[server_item.stop_event_id]

    with predict_job_lock:
        count_of_working_predict.value += 1

    time_line = _TimeLine()
    predict_count = 0
    while not stop_event.is_set():
        ready_memory_idx = ready_queue.get()
        time_line.record('get_ready')

        # Poison
        if isinstance(ready_memory_idx, _PoisonPill):
            poison_pill = ready_memory_idx
            # FIXME. tmp code
            all_success = False

            with predict_job_lock:
                # accumulate success predict_count
                poison_pill.predict_count += predict_count
                poison_pill.predict_count += predict_job_count.value

                # clear local predict_count
                predict_count = 0
                predict_job_count.value = 0

                # last process
                if count_of_working_predict.value == 1:
                    if poison_pill.predict_count == poison_pill.feed_count:  # all predict worker success
                        count_of_working_predict.value -= 1
                        logging.debug('pid={} write poison to complete queue'.
                                      format(os.getpid()))
                        all_success = True
                        # self._complete_queue.put(poison_pill)  # poison consumer  # NOTE! put here may hang
                    else:  # NOTE. some of predict worker failed
                        assert poison_pill.predict_count < poison_pill.feed_count,\
                            "if failed, predict_count={} must < feed_count={}".format(poison_pill.predict_count,
                                                                                      poison_pill.feed_count)
                        ready_queue.put(poison_pill)  # write back poison pill
                        continue  # continue predict failed job
                else:  # not last process
                    logging.debug('pid={} write poison back to ready'.format(
                        os.getpid()))
                    assert poison_pill.predict_count <= poison_pill.feed_count, \
                        "predict_count={} must <= feed_count={}".format(poison_pill.predict_count,
                                                                        poison_pill.feed_count)
                    count_of_working_predict.value -= 1
                    # self._ready_queue.put(poison_pill)  # poison other predict worker  # NOTE! put here may hang

            with predict_cond:
                # NOTE! see comment in _fetch_cond.wait()
                if all_success is True:
                    complete_queue.put(poison_pill)  # poison consumer
                else:
                    ready_queue.put(poison_pill)  # poison other predict worker

                # wait next reader iter or last failed predict job
                predict_cond.wait()

            with predict_job_lock:
                count_of_working_predict.value += 1
            continue

        logging.debug('pid={} ready_memory_idx={} memory_uid={}'.format(
            os.getpid(), ready_memory_idx,
            shared_memory_array.get_shared_uid(ready_memory_idx)))

        # Predict
        predict_success = d_batch_predict(shared_memory_array,
                                          ready_memory_idx, client)
        time_line.record('predict')

        # Failed
        if not predict_success:
            with predict_job_lock:
                predict_job_count.value += predict_count
                ready_queue.put(
                    ready_memory_idx)  # write back failed transaction
                # last process
                if count_of_working_predict.value == 1:
                    # NOTE. need notify other predict worker, or maybe deadlock
                    with predict_cond:
                        predict_cond.notify_all()

                count_of_working_predict.value -= 1
                predict_count = 0  # clear count
                return False

        logging.debug(
            'pid={} write ready_memory_idx={} memory_uid={} predict_count={}'.
            format(os.getpid(), ready_memory_idx,
                   shared_memory_array.get_shared_uid(ready_memory_idx),
                   predict_count))
        # predict complete
        complete_queue.put(ready_memory_idx)
        predict_count += 1
        time_line.record('put_complete')

    with predict_job_lock:
        predict_job_count.value += predict_count
        # last process
        if count_of_working_predict.value == 1:
            # FIXME. remove server, how to notify? if notify all, one process complete poison consumer
            # some other process may wait on the _ready_queue.get(), however this is ok for now.
            # NOTE. need notify other predict worker, or maybe deadlock
            with predict_cond:
                predict_cond.notify_all()
        count_of_working_predict.value -= 1
        predict_count = 0
    return True


def d_batch_predict(shared_memory_array, ready_memory_idx, client):
    d_batch_size, predict_feeds_memory, fetchs_memory = \
        shared_memory_array.get_predict_feeds_fetchs_memory(ready_memory_idx)

    # assert d_batch_size <= self.d_batch_size
    return client.predict(d_batch_size, predict_feeds_memory, fetchs_memory)


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
