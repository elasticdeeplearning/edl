# -*- coding: utf-8 -*-
import ast
import logging
import multiprocessing as mps
import numpy as np
import socket
import time
import threading

from contextlib import closing

from six.moves.configparser import ConfigParser
from six.moves import queue

from .parse_config import parse_serving_conf
from .shared_data import SharedMemoryArray, Var
from . import distill_worker

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


def is_server_alive(server):
    # FIXME. only for test, need find a better test method
    if distill_worker._NOP_PREDICT_TEST:
        return True
    alive = True
    ip, port = server.split(":")
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.settimeout(1.5)
            s.connect((ip, int(port)))
            s.shutdown(socket.SHUT_RDWR)
        except:
            alive = False
        return alive


class ServiceDiscover(object):
    def get_servers(self):
        pass


class FixedServiceDiscover(ServiceDiscover):
    def __init__(self, servers):
        assert isinstance(servers, (list, tuple, set))
        self._servers = servers

    def get_servers(self):
        return self._servers


class DynamicServiceDiscover(ServiceDiscover):
    def __init__(self, discovery_servers, require_num, service_name):
        self._discovery_servers = discovery_servers
        self._require_num = require_num
        self._service_name = service_name
        self._client = None

    def _connect(self):
        from paddle_edl.distill.discovery_client import DiscoveryClient
        client = DiscoveryClient(self._discovery_servers, self._service_name,
                                 self._require_num)
        client.start(daemon=True)
        self._client = client

    def get_servers(self):
        if self._client is None:
            self._connect()

        servers = self._client.get_servers()
        return servers

    def __del__(self):
        if self._client is not None:
            self._client.stop()
            self._client = None


_service_discover = None
_service_discover_lock = threading.Lock()


class DistillReader(object):
    def __init__(self):
        self.batch_size = None
        self.d_batch_size = 1
        self._capacity = 4
        self._occupied_capacity = None

        self._d_batch_reader = None

        self._serving_conf_file = None
        self._predict_feed_vars = dict()
        self._predict_fetch_vars = dict()

        self._reader_vars = []
        self._reader_fetch_names = []

        self._mode = None
        self._servers = []
        self._require_num = 1

        self._discovery_servers = []
        self._service_name = None

        # set shared memory conf
        self._max_thread = 2  # TODO. set max_thread
        # double buffer
        self._shared_len = 0
        # failed write back, is ok?
        self._queue_len = 0

        # get shared memory and idx queue
        self._shared_memory_array = None

        # FIXME. put these queue into shared_memory array?
        # idle--(feed)-->ready--(predict)-->complete--(fetch)-->idle
        self._idle_queue = None
        self._ready_queue = None
        self._complete_queue = None

        # complete memory--(multi copy)-->out memory
        # idle--(copy)-->ready--(yield)-->idle
        self._out_idle_queue = None
        self._out_ready_queue = None

        # work processor
        self._is_predict_start = False
        self._is_feed_start = False
        self._is_fetch_start = False
        self._predict_cond = mps.Condition()  # for function reentrant
        self._feed_cond = mps.Condition()  # for function reentrant
        self._fetch_cond = mps.Condition()  # for function reentrant

        # for predict worker to exit normally
        self._predict_job_lock = mps.Lock()
        self._count_of_working_predict = mps.Value('i', 0, lock=False)
        self._predict_job_count = mps.Value('i', 0, lock=False)

        # predict worker pool
        self._predict_manage_thread = None
        self._predict_stop_events = []
        self._predict_server_queue = None
        self._predict_server_result_queue = None

        self._is_init = False

    def set_batch_size(self, batch_size, teacher_batch_size=1):
        assert batch_size % teacher_batch_size == 0, \
                "teacher_batch_size must be able to divide batch_size."

        self.batch_size = batch_size
        self.d_batch_size = teacher_batch_size

    def set_capacity(self, capacity=4, occupied_capacity=0):
        assert capacity > occupied_capacity, "capacity must > occupied_capacity, or will hang"
        self._capacity = capacity
        self._occupied_capacity = occupied_capacity

    def load_serving_client_conf(self, conf_file):
        self._serving_conf_file = conf_file
        feed_vars, fetch_vars = parse_serving_conf(conf_file)
        self._predict_feed_vars = feed_vars
        self._predict_fetch_vars = fetch_vars

    def set_reader_feed_fetch(self, feed_names, feed_types, feed_shapes,
                              predict_fetch_names):
        self._reader_vars = []
        for name, dtype, shape in zip(feed_names, feed_types, feed_shapes):
            var = Var(name, dtype, shape)
            self._reader_vars.append(var)

        self._reader_fetch_names = predict_fetch_names

    def set_fixed_teacher(self, teachers):
        self._mode = 'fixed'
        self._servers = teachers
        self._require_num = len(teachers)

    def set_dynamic_teacher(self,
                            discovery_servers,
                            teacher_service_name,
                            require_max_teacher=1):
        self._mode = 'discover'
        self._discovery_servers = discovery_servers
        self._service_name = teacher_service_name
        self._require_num = require_max_teacher

    def init(self):
        assert self._is_init is False
        self._is_init = True

        batch_size = self.batch_size
        d_batch_size = self.d_batch_size

        # set shared memory conf
        self._max_thread = 2  # TODO. set max_thread
        # double buffer
        self._shared_len = max(2 * ((batch_size + d_batch_size - 1) /
                                    d_batch_size), 2 * self._require_num)
        # failed write back, is ok?
        self._queue_len = self._shared_len + self._require_num * self._max_thread

        # get shared memory and idx queue
        self._shared_memory_array = SharedMemoryArray(
            d_batch_size, batch_size, self._shared_len, self._capacity,
            self._predict_feed_vars, self._predict_fetch_vars,
            self._reader_vars, self._reader_fetch_names)

        # FIXME. put these queue into shared_memory array?
        # idle--(feed)-->ready--(predict)-->complete--(fetch)-->idle
        self._idle_queue = mps.Queue(self._queue_len)
        self._ready_queue = mps.Queue(self._queue_len)
        self._complete_queue = mps.Queue(self._queue_len)
        for i in range(self._shared_len):
            self._idle_queue.put(i)

        # complete memory--(multi copy)-->out memory
        # idle--(copy)-->ready--(yield)-->idle
        self._out_idle_queue = mps.Queue(self._capacity + 1)
        self._out_ready_queue = mps.Queue(self._capacity + 1)
        for i in range(self._capacity):
            self._out_idle_queue.put(i)

        # work processor
        # self._is_predict_start = False
        # self._is_feed_start = False
        # self._is_fetch_start = False
        # self._predict_cond = mps.Condition()  # for function reentrant
        # self._feed_cond = mps.Condition()  # for function reentrant
        # self._fetch_cond = mps.Condition()  # for function reentrant

        # for predict worker to exit normally
        # self._predict_job_lock = mps.Lock()
        # self._count_of_working_predict = mps.Value('i', 0, lock=False)
        # self._predict_job_count = mps.Value('i', 0, lock=False)

        # predict worker pool
        self._predict_manage_thread = None
        self._predict_stop_events = [
            mps.Event() for i in range(self._require_num)
        ]
        self._predict_server_queue = mps.Queue(self._require_num)
        self._predict_server_result_queue = mps.Queue(self._require_num)

    def _get_servers(self, first_in):
        global _service_discover
        if _service_discover is not None:
            return _service_discover.get_servers()

        # FIXME. The order of object destruction
        if not first_in:
            logging.warning('service discover must have been deconstructed')
            return None

        with _service_discover_lock:
            assert first_in is True, 'service discover must be create in first in'
            while _service_discover is None:
                if self._mode == 'fixed':
                    _service_discover = FixedServiceDiscover(self._servers)
                elif self._mode == 'discover':
                    _service_discover = DynamicServiceDiscover(
                        self._discovery_servers, self._require_num,
                        self._service_name)
                else:
                    raise TypeError('mode must be fixed or discover')
        return _service_discover.get_servers()

    def _predict_manage_worker(self, process):
        num_shutdown_process = [0]

        def shutdown_one_process():
            self._predict_server_queue.put(None)
            num_shutdown_process[0] += 1

        server_id = 0  # not yet used
        server_to_item = dict()
        idle_predict_num = self._require_num
        event_set = set()
        for i in range(self._require_num):
            event_set.add(i)

        # Fix the order of object destruction
        first_in = True
        while True:
            try:
                # server job stop, return back stop_event_id
                server_result_item = self._predict_server_result_queue.get(
                    block=False)
                stop_event_id = server_result_item.stop_event_id
                event_set.add(stop_event_id)
                del server_to_item[server_result_item.server]

                # clear event
                self._predict_stop_events[stop_event_id].clear()
                # directly use count
                idle_predict_num += 1
                assert idle_predict_num <= self._require_num, \
                    'idle_predict_num={} must <= require_num={}'.format(
                        idle_predict_num, self._require_num)
                logging.info('Removed server={}'.format(
                    server_result_item.server))
            except queue.Empty:
                pass

            servers = self._get_servers(first_in)
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
                if not self._predict_stop_events[stop_event_id].is_set():
                    self._predict_stop_events[stop_event_id].set()
                    logging.info('Removing server={}'.format(server))

            # Add servers
            while len(add_servers) != 0:
                if idle_predict_num == 0:
                    break
                assert idle_predict_num > 0, 'idle_predict_num must > 0'

                server = add_servers.pop()
                if not is_server_alive(server):
                    logging.warning('server={} is not alive'.format(server))
                    continue

                idle_predict_num -= 1
                event_id = event_set.pop()
                server_item = distill_worker._ServerItem(server_id, server,
                                                         event_id)
                self._predict_server_queue.put(server_item)
                server_to_item[server] = server_item
                server_id += 1
                logging.info('Adding server={}'.format(server))

            time.sleep(1.5)

    def _start_feed_worker(self, reader):
        if not self._is_feed_start:
            feed_worker = mps.Process(
                target=distill_worker.feed_worker,
                args=(
                    reader,
                    self._idle_queue,
                    self._ready_queue,
                    self._shared_memory_array,
                    self._feed_cond, ))
            feed_worker.daemon = True
            feed_worker.start()
            self._is_feed_start = True
        else:
            with self._feed_cond:
                self._feed_cond.notify()

    def _start_predict_worker(self):
        process = []
        for i in range(self._require_num):
            stop_flag = mps.Event()
            worker = mps.Process(
                target=distill_worker.predict_worker,
                args=(
                    self._predict_server_queue,
                    self._predict_server_result_queue,
                    self._shared_memory_array,
                    self._serving_conf_file,
                    self._predict_stop_events,
                    self._predict_job_lock,
                    self._count_of_working_predict,
                    self._ready_queue,
                    self._predict_job_count,
                    self._predict_cond,
                    self._complete_queue, ))
            worker.daemon = True
            worker.start()
            process.append(worker)
        return process

    def _start_predict_worker_pool(self):
        if not self._is_predict_start:
            # start predict worker pool
            process = self._start_predict_worker()
            self._predict_manage_thread = threading.Thread(
                target=self._predict_manage_worker, args=(process, ))
            self._predict_manage_thread.daemon = True
            self._predict_manage_thread.start()

            self._is_predict_start = True
        else:
            # wake up predict worker pool
            with self._predict_cond:
                self._predict_cond.notify_all()

    def _start_fetch_worker(self):
        if not self._is_fetch_start:
            fetch_worker = mps.Process(
                target=distill_worker.fetch_worker,
                args=(
                    self.batch_size,
                    self.d_batch_size,
                    self._shared_memory_array,
                    self._idle_queue,
                    self._complete_queue,
                    self._out_idle_queue,
                    self._out_ready_queue,
                    self._fetch_cond, ))
            fetch_worker.daemon = True
            fetch_worker.start()
            self._is_fetch_start = True
        else:
            with self._fetch_cond:
                self._fetch_cond.notify()

    def _set_d_batch_generator(self, reader):
        """ reader format: type(list)([np_slot0, np_slot1, ...] """
        self._d_batch_reader = reader
        return self

    def set_sample_generator(self, reader):
        """ reader format: type(list|tuple)([slot0, slot1, slot3]) """
        assert self._d_batch_reader is None, "can't reset reader"
        feed_count = self._shared_memory_array.get_feed_count()

        # FIXME. how to support lod tensor?

        def __d_batch_reader_impl__():
            samples_length = 0
            slots = [[] for i in range(feed_count)]

            for sample in reader():
                assert len(sample) == feed_count, \
                    "The number of feeds in the sample is not equal with the number of feeds in config file"
                for i, slot in enumerate(sample):
                    slots[i].append(slot)
                samples_length += 1

                if samples_length == self.d_batch_size:
                    d_batch = []
                    for d_item in slots:
                        d_batch.append(np.array(d_item))
                    yield d_batch
                    samples_length = 0
                    slots = [[] for i in range(feed_count)]

            if samples_length != 0:
                d_batch = []
                for d_item in slots:
                    d_batch.append(np.array(d_item))
                yield d_batch

        self._set_d_batch_generator(__d_batch_reader_impl__)
        return self

    def set_sample_list_generator(self, reader):
        """ reader format: type(list)([(sp0_slot0, sp0_slot1, ..), (sp1_slot0, ..), ...])"""
        assert self._d_batch_reader is None, "can't reset reader"
        feed_count = self._shared_memory_array.get_feed_count()

        def __d_batch_reader_impl__():
            samples_length = 0
            slots = [[] for i in range(feed_count)]

            for samples in reader():
                for sample in samples:
                    assert len(sample) == feed_count, \
                        "The number of feeds in the sample is not equal with the number of feeds in config file"
                    for i, slot in enumerate(sample):
                        slots[i].append(slot)
                    samples_length += 1

                    if samples_length == self.d_batch_size:
                        d_batch = []
                        for d_item in slots:
                            d_batch.append(np.array(d_item))
                        yield d_batch
                        samples_length = 0
                        slots = [[] for i in range(feed_count)]

            if samples_length != 0:
                d_batch = []
                for d_item in slots:
                    d_batch.append(np.array(d_item))
                yield d_batch

        self._set_d_batch_generator(__d_batch_reader_impl__)
        return self

    def set_batch_generator(self, reader):
        """ reader format: type(list)([np_slot0, np_slot1, ...]) """
        assert self._d_batch_reader is None, "can't reset reader"
        feed_count = self._shared_memory_array.get_feed_count()

        def __d_batch_reader_impl__():
            for samples in reader():
                # TODO
                pass

        raise NotImplementedError("no implmented")

    def distill_reader(self):
        def __reader_creator__():
            # NOTE. qsize will Raises NotImplementedError on Mac OSX because of broken sem_getvalue()
            # assert self._idle_queue.qsize() == self._shared_len
            # assert self._out_idle_queue.qsize() == self._capacity
            assert self._ready_queue.empty()
            assert self._complete_queue.empty()
            assert self._out_ready_queue.empty()
            assert self._d_batch_reader is not None, "must set reader before iter distill_reader"

            self._start_feed_worker(self._d_batch_reader)
            self._start_fetch_worker()
            # NOTE. When using logging, if start_predict_worker_pool is before start_feed_worker
            # or start_fetch_worker, logging maybe hang!!!
            # For specific reasons, please see https://bugs.python.org/issue6721
            # https://stackoverflow.com/questions/24509650/deadlock-with-logging-multiprocess-multithread-python-script
            # >>> The problem is common in any situation where you have locks, threads and forks.
            # >>> If thread 1 had a lock while thread 2 calls fork, in the forked process,
            # >>> there will only be thread 2 and the lock will be held forever.
            # So need to move start_predict_worker_pool to the end if we use logging in predict
            # manager thread, or for the sake of safety, don't use logging?
            self._start_predict_worker_pool()

            # numpy array share memory with output, when outside is using array
            # we can't reuse it. Otherwise, dirty data may be read by outside.
            idx_being_used = []
            while True:
                idx = self._out_ready_queue.get()
                if idx == distill_worker.POISON_PILL:
                    break
                out_np = self._shared_memory_array.get_np_from_out(idx)
                idx_being_used.append(idx)
                yield out_np
                #self._out_idle_queue.put(idx)
                # FIXME. is right?
                if len(idx_being_used) > self._occupied_capacity:
                    self._out_idle_queue.put(idx_being_used[0])
                    idx_being_used = idx_being_used[1:]

            for i in idx_being_used:
                self._out_idle_queue.put(i)

        return __reader_creator__


class FakeDistillReader(object):
    def __init__(self, conf_file):
        config = ConfigParser()
        config.read(conf_file)
        # fetch vars conf
        self._fetch_vars = ast.literal_eval(config.get('fetch', 'fetch_vars'))
        self._fetch_types = ast.literal_eval(
            config.get('fetch', 'fetch_types'))
        self._fetch_shapes = ast.literal_eval(
            config.get('fetch', 'fetch_shapes'))

    def fake_from_sample_generator(self, reader):
        """ reader format: type(list|tuple)([slot0, slot1, slot3]) """
        fetchs = []
        for i, var in enumerate(self._fetch_vars):
            tmp_var = np.zeros(
                shape=self._fetch_shapes[i], dtype=self._fetch_types[i])
            fetchs.append(tmp_var)

        def __fake_reader_impl__():
            for sample in reader():
                if type(sample) is tuple:
                    yield tuple(list(sample) + fetchs)
                elif type(sample) is list:
                    yield sample + fetchs
                else:
                    raise TypeError('reader sample must be tuple or list')

        return __fake_reader_impl__

    def fake_from_sample_list_generator(self, reader):
        """ reader format: type(list)([(sp0_slot0, sp0_slot1, ..), (sp1_slot0, ..), ...])"""
        fetchs = []
        for i, var in enumerate(self._fetch_vars):
            tmp_var = np.zeros(
                shape=self._fetch_shapes[i], dtype=self._fetch_types[i])
            fetchs.append(tmp_var)

        def __fake_reader_impl__():
            for samples in reader():
                new_samples = []
                for sample in samples:
                    if type(sample) is tuple:
                        new_samples.append(tuple(list(sample) + fetchs))
                    elif type(sample) is list:
                        new_samples.append(sample + fetchs)
                    else:
                        raise TypeError('reader sample must be tuple or list')
                yield new_samples

        return __fake_reader_impl__
