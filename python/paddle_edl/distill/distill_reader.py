# -*- coding: utf-8 -*-
import ast
import ctypes
import logging
import multiprocessing as mps
import numpy as np
import os
import socket
import sys
import time
import threading

from contextlib import closing
from paddle_serving_client import Client

from six.moves.configparser import ConfigParser
from six.moves import reduce
from six.moves import queue

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")

# only for local test.
_NOP_PREDICT_TEST = False


class _NopTimeLine(object):
    def record(self, name):
        pass

    def reset(self):
        pass


class _RealTimeLine(object):
    def __init__(self):
        self.pid = os.getpid()
        self.time = time.time()

    def record(self, name):
        new_time = time.time()
        sys.stderr.write('pid={} op={} time={}ms\n'.format(self.pid, name, (
            new_time - self.time) * 1000))
        self.time = new_time

    def reset(self):
        self.time = time.time()


_is_profile = int(os.environ.get('DISTILL_READER_PROFILE', 0))
_TimeLine = _RealTimeLine if _is_profile else _NopTimeLine


def is_server_alive(server):
    # FIXME. only for test, need find a better test method
    if _NOP_PREDICT_TEST:
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


class _Var(object):
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = shape

        self.ctype = self.dtype2ctype(dtype)
        self.numel = reduce(lambda x, y: x * y, self.shape)
        # array type
        self.atype = self.numel * self.ctype

    def __str__(self):
        return "{{name: '{}', dtype: '{}', shape: {}, numel: {}, atype: {}}}".format(
            self.name, self.dtype, self.shape, self.numel, self.atype)

    @staticmethod
    def dtype2ctype(dtype):
        if dtype == 'float32':
            return ctypes.c_float
        elif dtype == 'int64':
            return ctypes.c_int64
        elif dtype == 'int32':
            return ctypes.c_int32
        assert False, 'now unsupport dtype={}'.format(dtype)


class _DataStruct(object):
    def __init__(self, config, d_batch_size, batch_size):
        # feed vars conf
        self._feed_vars = ast.literal_eval(config.get('feed', 'feed_vars'))
        self._feed_types = ast.literal_eval(config.get('feed', 'feed_types'))
        self._feed_shapes = ast.literal_eval(config.get('feed', 'feed_shapes'))
        self._predict_feed_ids = ast.literal_eval(
            config.get('feed', 'predict_feed_ids'))
        logging.info((self._feed_vars, self._feed_types, self._feed_shapes,
                      self._predict_feed_ids))

        # fetch vars conf
        self._fetch_vars = ast.literal_eval(config.get('fetch', 'fetch_vars'))
        self._fetch_types = ast.literal_eval(
            config.get('fetch', 'fetch_types'))
        self._fetch_shapes = ast.literal_eval(
            config.get('fetch', 'fetch_shapes'))
        self._fetch_ids = [
            i + len(self._feed_vars) for i in range(len(self._fetch_vars))
        ]
        logging.info((self._fetch_vars, self._fetch_types, self._fetch_shapes,
                      self._fetch_ids))

        # get all_vars and DataStruct
        self._all_vars = []

        def get_all_vars(_vars, types, shapes):
            for _var, var_type, var_shape in zip(_vars, types, shapes):
                self._all_vars.append(_Var(_var, var_type, var_shape))

        get_all_vars(self._feed_vars, self._feed_types, self._feed_shapes)
        get_all_vars(self._fetch_vars, self._fetch_types, self._fetch_shapes)
        logging.info([str(s) for s in self._all_vars])

        def get_uid(names):
            uid = '_uid_'
            idx = 0
            ret = uid
            while ret in names:
                ret = uid + str(idx)
                idx += 1
            return ret

        # uid contains d_batch uid&len
        self._uid = get_uid([var.name for var in self._all_vars])
        data_struct = [(self._uid, ctypes.c_uint64 * 2)]
        out_struct = [(self._uid, ctypes.c_uint64 * 2)]

        for var in self._all_vars:
            data_struct.append((var.name, d_batch_size * var.atype))
            out_struct.append((var.name, batch_size * var.atype))
        logging.info('data_struct = ' + str(data_struct))
        logging.info('out_struct = ' + str(out_struct))

        class DataStruct(ctypes.Structure):
            """
            DataStruct {
                '_uid_': int64[2], # uid, filled length
                'feed_var0': .dtype[d_batch_size][.numel],
                ...,
                'fetch_var0': .dtype[d_batch_size][.numel],
                ...,
            }
            """
            _fields_ = data_struct

        class OutStruct(ctypes.Structure):
            _fields_ = out_struct

        self.DataStruct = DataStruct
        self.OutStruct = OutStruct
        self.d_batch_size = d_batch_size
        self.batch_size = batch_size

        logging.info('DataStruct size={} Byte'.format(
            ctypes.sizeof(self.DataStruct)))
        logging.debug('DataStruct.{}={}'.format(self._all_vars[
            -1].name, getattr(DataStruct, self._all_vars[-1].name)))

    def get_vars(self):
        return self._all_vars

    def get_fetchs_name(self):
        return self._fetch_vars

    def get_feed_name(self, idx):
        return self._all_vars[idx].name

    def get_predict_feeds_name(self):
        return [self._all_vars[idx].name for idx in self._predict_feed_ids]

    def get_predict_feeds(self):
        return [self._all_vars[idx] for idx in self._predict_feed_ids]

    def get_fetchs(self):
        return [self._all_vars[idx] for idx in self._fetch_ids]

    def get_feed_count(self):
        return len(self._feed_vars)

    def get_uid_name(self):
        return self._uid

    def get_vars_name(self):
        return [var.name for var in self._all_vars]


class _SharedMemoryArray(object):
    def __init__(self, config, d_batch_size, batch_size, shared_len, capacity):
        self._d_batch_size = d_batch_size
        self._batch_size = batch_size
        self._data_struct = _DataStruct(config, d_batch_size, batch_size)

        self._shared_len = shared_len
        self._capacity = capacity

        # --- get shared memory ---
        self._shared_list = []
        self._out_list = []
        logging.warning('Alloc shared memory size={}MiB'.format(
            ctypes.sizeof(self._data_struct.DataStruct) * self._shared_len / (
                1024. * 1024.)))
        for i in range(self._shared_len):
            self._shared_list.append(
                mps.Value(
                    self._data_struct.DataStruct, lock=False))

        logging.warning('Alloc out shared memory size={}MiB'.format(
            ctypes.sizeof(self._data_struct.OutStruct) * self._capacity / (
                1024. * 1024.)))
        for i in range(capacity):
            self._out_list.append(
                mps.Value(
                    self._data_struct.OutStruct, lock=False))

    def get_uid_name(self):
        return self._data_struct.get_uid_name()

    def get_fetchs_name(self):
        return self._data_struct.get_fetchs_name()

    def get_predict_feeds_name(self):
        return self._data_struct.get_predict_feeds_name()

    def get_predict_feeds(self):
        return self._data_struct.get_predict_feeds()

    def get_fetchs(self):
        return self._data_struct.get_fetchs()

    def _get_feed_name(self, idx):
        return self._data_struct.get_feed_name(idx)

    def get_feed_count(self):
        return self._data_struct.get_feed_count()

    def get_shared_memory(self, memory_idx, name=None, offset=None):
        """ 获取第memory_idx个共享内存，name字段，第offset个数组的memory """
        if name is None and offset is None:
            return self._shared_list[memory_idx]
        elif offset is None:
            return getattr(self._shared_list[memory_idx], name)
        elif name is None:
            raise Exception('TODO add get memory_idx, offset')
        else:
            return getattr(self._shared_list[memory_idx], name)[offset]

    def get_shared_uid(self, memory_idx):
        return self.get_shared_memory(memory_idx, self.get_uid_name())[:]

    def _get_feed_memory(self, memory_idx, feed_idx, offset=None):
        ''' 获取第memory_idx个共享内存，第feed_idx个feed，第offset个数组的memory '''
        name = self._get_feed_name(feed_idx)
        return self.get_shared_memory(memory_idx, name, offset)

    def get_predict_feeds_fetchs_memory(self, memory_idx):
        """ 获取memory_idx处共享内存的d_batch_size, 预测输入内存, 预测输出内存"""
        uid, d_batch_size = self.get_shared_uid(memory_idx)

        predict_feeds_name = self.get_predict_feeds_name()
        predict_feeds_memory = [
            self.get_shared_memory(memory_idx, name)
            for name in predict_feeds_name
        ]

        fetchs_name = self.get_fetchs_name()
        fetchs_memory = [
            self.get_shared_memory(memory_idx, name) for name in fetchs_name
        ]
        return d_batch_size, predict_feeds_memory, fetchs_memory

    def _write_uid(self, memory_idx, uid, offset):
        memory = self.get_shared_memory(memory_idx, self.get_uid_name())
        memory[0], memory[1] = uid, offset

    def _write_feed_slot(self, memory_idx, feed_idx, offset, slot):
        ''' 将feed slot写入第memory_idx个共享内存，第feed_idx个feed，第offset个数组的memory中'''
        memory = self._get_feed_memory(memory_idx, feed_idx, offset)
        assert ctypes.sizeof(memory) == slot.nbytes, \
            "feed bytes={} != memory bytes={}".format(slot.nbytes, ctypes.sizeof(memory))
        ctypes.memmove(memory, slot.ctypes.data, slot.nbytes)
        # logging.debug('write feed, memory_idx={}, feed_idx={}, offset={},\nmemory={}'.format(
        #     memory_idx, feed_idx, offset, list(memory)
        # ))

    def write_feed_slots(self, memory_idx, uid, offset, slots):
        ''' 将feed slots写入第memory_idx个共享内存的feed变量的第offset个位置，如果填充满，返回true'''
        for feed_idx, slot in enumerate(slots):
            assert isinstance(slot, np.ndarray), 'feed format is wrong'
            self._write_feed_slot(memory_idx, feed_idx, offset, slot)

        self._write_uid(memory_idx, uid, offset + 1)
        if offset + 1 == self._d_batch_size:
            return True
        return False

    def _write_feed_items(self, memory_idx, feed_idx, items, d_batch_size):
        """ 将items写入到第memory_idx个共享内存的第feed_idx个feed中 """
        memory = self._get_feed_memory(memory_idx, feed_idx)
        assert ctypes.sizeof(memory[0]) * d_batch_size == items.nbytes, \
            "feed items bytes={} != memory bytes={}".format(items.nbytes, ctypes.sizeof(memory))
        ctypes.memmove(memory, items.ctypes.data, items.nbytes)
        # logging.debug('write feed, memory_idx={}, feed_idx={}, \nmemory={}'.format(
        #     memory_idx, feed_idx, list(memory[d_batch_size-1])
        # ))

    def write_feed(self, memory_idx, uid, d_batch):
        """ 将d_batch样本写入到第memory_idx个共享内存的feed中 """
        # full d_batch_size
        d_batch_size = d_batch[0].shape[0]
        assert d_batch_size <= self._d_batch_size, \
            "Something wrong: feed d_batch_size={} must <= conf d_batch_size={}".format(
                d_batch_size, self._d_batch_size)
        for feed_idx, items in enumerate(d_batch):
            self._write_feed_items(memory_idx, feed_idx, items, d_batch_size)

        self._write_uid(memory_idx, uid, d_batch_size)

    def get_out_memory(self, out_idx, name=None, offset=None):
        """ 获取第out_idx个out共享内存，name字段，第offset个数组的memory """
        if name is None and offset is None:
            return self._out_list[out_idx]
        elif offset is None:
            return getattr(self._out_list[out_idx], name)
        elif name is None:
            raise Exception('TODO add get memory_idx, offset')
        else:
            return getattr(self._out_list[out_idx], name)[offset]

    def copy_shm_to_out(self, src_memory_idx, out_idx, out_offset, out_uid,
                        copy_batch_size):
        """ 将predict的src_memory_idx处的memory拷贝到 output的memory中，该memory在out_idx偏移为out_offset处 """
        for name in self._data_struct.get_vars_name():
            out_memory = self.get_out_memory(out_idx, name, out_offset)
            src_memory = self.get_shared_memory(src_memory_idx, name)
            ctypes.memmove(out_memory, src_memory, ctypes.sizeof(src_memory))

        out_uid_memory = self.get_out_memory(out_idx, self.get_uid_name())
        out_uid_memory[0], out_uid_memory[
            1] = out_uid, out_offset + copy_batch_size

    def get_np_from_out(self, out_idx):
        """ 将output位于out_idx的内存转为numpy
        NOTO. 特别注意！numpy数组共享output的内存，所以在被外部使用时，需要标记为占用，
        等外部使用完，才可以再循环使用，这也即是添加idx_being_used的原因。
        """
        numpy_list = []
        uid, batch_size = self.get_out_memory(out_idx, self.get_uid_name())[:]

        for var in self._data_struct.get_vars():
            out_memory = self.get_out_memory(out_idx, var.name)
            out_array = np.frombuffer(out_memory, var.dtype, batch_size * var.numel)\
                .reshape([int(batch_size)] + list(var.shape))
            logging.debug('var={} shape={}'.format(
                var.name, [int(batch_size)] + list(var.shape)))
            numpy_list.append(out_array)
        return numpy_list


class _ServerItem(object):
    PENDING = 'pending'
    ERROR = 'error'
    FINISHED = 'finished'

    def __init__(self, server_id, server, stop_event_id, state=PENDING):
        self.server_id = server_id
        self.server = server
        self.stop_event_id = stop_event_id
        self.state = state


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
    def __init__(self, host, port, require_num, service_name):
        self._host = host
        self._port = port
        self._require_num = require_num
        self._service_name = service_name
        self._client = None

    def _connect(self):
        from paddle_edl.distill.discovery_client import DiscoveryClient
        client = DiscoveryClient(['{}:{}'.format(self._host, self._port)],
                                 self._service_name, self._require_num)
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


class _PoisonPill:
    def __init__(self, feed_count, predict_count=0, complete_count=0):
        self.feed_count = feed_count
        self.predict_count = predict_count
        self.complete_count = complete_count


class DistillReader(object):
    POISON_PILL = -1

    def __init__(self, conf_file, batch_size, d_batch_size, capacity,
                 occupied_capacity):
        assert batch_size % d_batch_size == 0, \
                "distill_batch_size must be able to divide batch_size."
        assert capacity > occupied_capacity, "capacity must > occupied_capacity, or will hang"
        self.batch_size = batch_size
        self.d_batch_size = d_batch_size
        self._capacity = capacity
        self._occupied_capacity = occupied_capacity

        self._d_batch_reader = None

        config = ConfigParser()
        config.read(conf_file)

        self._mode = config.get('conf', 'mode')
        self._serving_conf_file = config.get('conf', 'serving_conf_file')
        if self._mode == 'fixed':
            # fixed servers conf
            self._servers = ast.literal_eval(config.get('conf', 'servers'))
            self._require_num = len(self._servers)
            logging.info((self._mode, self._servers, self._require_num))
        elif self._mode == 'discover':
            # discovery service conf
            self._host = config.get('conf', 'host')
            self._port = config.getint('conf', 'port')
            self._service_name = config.get('conf', 'service_name')
            self._require_num = config.getint('conf', 'require_num')
            logging.info((self._host, self._port, self._service_name,
                          self._require_num))

        # set shared memory conf
        self._max_thread = 2  # TODO. set max_thread
        # double buffer
        self._shared_len = max(2 * ((batch_size + d_batch_size - 1) /
                                    d_batch_size), 2 * self._require_num)
        # failed write back, is ok?
        self._queue_len = self._shared_len + self._require_num * self._max_thread

        # get shared memory and idx queue
        self._shared_memory_array = _SharedMemoryArray(
            config, d_batch_size, batch_size, self._shared_len, self._capacity)

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
        #self._idle_predict_semaphore = mps.Semaphore(self._require_num)
        self._predict_stop_events = [
            mps.Event() for i in range(self._require_num)
        ]
        self._predict_server_queue = mps.Queue(self._require_num)
        self._predict_server_result_queue = mps.Queue(self._require_num)

    def _feed_worker(self, reader):
        """ supported samples format [np_var0, np_var1, ..] each numpy contains <=d_batch_size """
        logging.info('feed_work pid{}'.format(os.getpid()))
        # TODO. exit feed_worker
        while True:
            uid = 0
            for d_batch in reader():
                idle_memory_idx = self._idle_queue.get()
                self._shared_memory_array.write_feed(idle_memory_idx, uid,
                                                     d_batch)
                self._ready_queue.put(idle_memory_idx)
                uid += 1

            poison_pill = _PoisonPill(uid)

            # self._ready_queue.put(poison_pill)  # NOTE! can't put here, or will hang
            with self._feed_cond:
                # NOTE! see comment in _fetch_cond.wait()
                self._ready_queue.put(poison_pill)
                self._feed_cond.wait()

    def _d_batch_predict(self, ready_memory_idx, client):

        d_batch_size, predict_feeds_memory, fetchs_memory = \
            self._shared_memory_array.get_predict_feeds_fetchs_memory(ready_memory_idx)

        assert d_batch_size <= self.d_batch_size
        return client.predict(d_batch_size, predict_feeds_memory,
                              fetchs_memory)

    def _predict(self, server_item):
        feeds = self._shared_memory_array.get_predict_feeds()
        fetchs = self._shared_memory_array.get_fetchs()

        logging.info('connect server={}'.format(server_item.server))
        predict_server = PaddlePredictServer if _NOP_PREDICT_TEST is False else _TestNopPaddlePredictServer
        client = predict_server(server_item.server, self._serving_conf_file,
                                feeds, fetchs)
        if client.connect() is False:
            return False

        stop_event = self._predict_stop_events[server_item.stop_event_id]

        with self._predict_job_lock:
            self._count_of_working_predict.value += 1

        time_line = _TimeLine()
        predict_count = 0
        while not stop_event.is_set():
            ready_memory_idx = self._ready_queue.get()
            time_line.record('get_ready')

            # Poison
            if isinstance(ready_memory_idx, _PoisonPill):
                poison_pill = ready_memory_idx
                # FIXME. tmp code
                all_success = False

                with self._predict_job_lock:
                    # accumulate success predict_count
                    poison_pill.predict_count += predict_count
                    poison_pill.predict_count += self._predict_job_count.value

                    # clear local predict_count
                    predict_count = 0
                    self._predict_job_count.value = 0

                    # last process
                    if self._count_of_working_predict.value == 1:
                        if poison_pill.predict_count == poison_pill.feed_count:  # all predict worker success
                            self._count_of_working_predict.value -= 1
                            logging.debug(
                                'pid={} write poison to complete queue'.format(
                                    os.getpid()))
                            all_success = True
                            # self._complete_queue.put(poison_pill)  # poison consumer  # NOTE! put here may hang
                        else:  # NOTE. some of predict worker failed
                            assert poison_pill.predict_count < poison_pill.feed_count,\
                                "if failed, predict_count={} must < feed_count={}".format(poison_pill.predict_count,
                                                                                          poison_pill.feed_count)
                            self._ready_queue.put(
                                poison_pill)  # write back poison pill
                            continue  # continue predict failed job
                    else:  # not last process
                        logging.debug('pid={} write poison back to ready'.
                                      format(os.getpid()))
                        assert poison_pill.predict_count <= poison_pill.feed_count, \
                            "predict_count={} must <= feed_count={}".format(poison_pill.predict_count,
                                                                            poison_pill.feed_count)
                        self._count_of_working_predict.value -= 1
                        # self._ready_queue.put(poison_pill)  # poison other predict worker  # NOTE! put here may hang

                with self._predict_cond:
                    # NOTE! see comment in _fetch_cond.wait()
                    if all_success is True:
                        self._complete_queue.put(
                            poison_pill)  # poison consumer
                    else:
                        self._ready_queue.put(
                            poison_pill)  # poison other predict worker

                    # wait next reader iter or last failed predict job
                    self._predict_cond.wait()

                with self._predict_job_lock:
                    self._count_of_working_predict.value += 1
                continue

            logging.debug('pid={} ready_memory_idx={} memory_uid={}'.format(
                os.getpid(), ready_memory_idx,
                self._shared_memory_array.get_shared_uid(ready_memory_idx)))

            # Predict
            predict_success = self._d_batch_predict(ready_memory_idx, client)
            time_line.record('predict')

            # Failed
            if not predict_success:
                with self._predict_job_lock:
                    self._predict_job_count.value += predict_count
                    self._ready_queue.put(
                        ready_memory_idx)  # write back failed transaction
                    # last process
                    if self._count_of_working_predict.value == 1:
                        # NOTE. need notify other predict worker, or maybe deadlock
                        with self._predict_cond:
                            self._predict_cond.notify_all()

                    self._count_of_working_predict.value -= 1
                    predict_count = 0  # clear count
                    return False

            logging.debug(
                'pid={} write ready_memory_idx={} memory_uid={} predict_count={}'.
                format(os.getpid(), ready_memory_idx,
                       self._shared_memory_array.get_shared_uid(
                           ready_memory_idx), predict_count))
            # predict complete
            self._complete_queue.put(ready_memory_idx)
            predict_count += 1
            time_line.record('put_complete')

        with self._predict_job_lock:
            self._predict_job_count.value += predict_count
            # last process
            if self._count_of_working_predict.value == 1:
                # FIXME. remove server, how to notify? if notify all, one process complete poison consumer
                # some other process may wait on the _ready_queue.get(), however this is ok for now.
                # NOTE. need notify other predict worker, or maybe deadlock
                with self._predict_cond:
                    self._predict_cond.notify_all()
            self._count_of_working_predict.value -= 1
            predict_count = 0
        return True

    def _predict_worker(self):
        logging.info('predict_worker pid={}'.format(os.getpid()))
        while True:
            # get server item
            server_item = self._predict_server_queue.get(block=True)
            if server_item is None:
                self._predict_server_result_queue.put(None)
                return

            # predict
            is_normal_stop = self._predict(server_item)

            server_item.state = _ServerItem.FINISHED \
                if is_normal_stop else _ServerItem.ERROR

            # clear event, return result, release semaphore
            # self._predict_stop_events[server_item.stop_event_id].clear()
            self._predict_server_result_queue.put(server_item)
            #self._idle_predict_semaphore.release()
            logging.info('Stopped server={}'.format(server_item.server))

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
                        self._host, self._port, self._require_num,
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
                server_item = _ServerItem(server_id, server, event_id)
                self._predict_server_queue.put(server_item)
                server_to_item[server] = server_item
                server_id += 1
                logging.info('Adding server={}'.format(server))

            time.sleep(1.5)

    def _fetch_worker(self):
        logging.info('fetch_worker pid={}'.format(os.getpid()))
        # init
        out_idx = self._out_idle_queue.get()
        assert out_idx == 0, "first out_idx must == 0"
        out_offset = 0
        last_complete_memory_idx, last_length = -1, -1

        out_uid = 0
        complete_count = 0
        while True:
            complete_memory_idx = self._complete_queue.get()

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
                    self._shared_memory_array.copy_shm_to_out(
                        last_complete_memory_idx, out_idx, out_offset, out_uid,
                        last_length)
                    self._idle_queue.put(last_complete_memory_idx)

                    last_complete_memory_idx = -1
                    out_offset += last_length

                    poison_pill.complete_count += 1

                if poison_pill.complete_count == poison_pill.predict_count:  # all fetch job success
                    assert last_complete_memory_idx == -1, "no full batch fetch failed?"
                    if out_offset > 0:
                        # put last no full ready data to out_ready_queue
                        self._out_ready_queue.put(out_idx)
                    else:
                        # write back idle out idx
                        self._out_idle_queue.put(out_idx)

                    # poison distill reader
                    # self._out_ready_queue.put(DistillReader.POISON_PILL)  # NOTE. POISON here may be hang
                    with self._fetch_cond:
                        # NOTE!!! put poison pill can only be placed inside the critical area of the cond variable,
                        # otherwise, when reader finishes, reentrant and sending notify,
                        # the current process may not have entered the critical area and cannot receive notifications.
                        # The will hang.
                        self._out_ready_queue.put(DistillReader.POISON_PILL)
                        self._fetch_cond.wait()

                    # init
                    out_idx = self._out_idle_queue.get()
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
                    self._complete_queue.put(poison_pill)  # write back poison
                    continue

            uid_list = self._shared_memory_array.get_shared_uid(
                complete_memory_idx)
            uid, filled_length = uid_list
            logging.debug('result_memory_idx={} memory_uid={}'.format(
                complete_memory_idx, uid_list))

            assert filled_length <= self.d_batch_size
            # not full, must be last batch
            if filled_length != self.d_batch_size:
                logging.debug('filled_length={} d_batch_size={} idx={}'.format(
                    filled_length, self.d_batch_size, complete_memory_idx))
                last_complete_memory_idx = complete_memory_idx
                last_length = filled_length
            else:  # full
                self._shared_memory_array.copy_shm_to_out(
                    complete_memory_idx, out_idx, out_offset, out_uid,
                    self.d_batch_size)
                # copy done, write back idx to idle queue
                self._idle_queue.put(complete_memory_idx)
                complete_count += 1

                out_offset += self.d_batch_size
                # out is ready
                if out_offset == self.batch_size:
                    self._out_ready_queue.put(out_idx)
                    out_idx = self._out_idle_queue.get()
                    out_uid += 1
                    out_offset = 0

    def _start_feed_worker(self, reader):
        if not self._is_feed_start:
            feed_worker = mps.Process(
                target=self._feed_worker, args=(reader, ))
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
            worker = mps.Process(target=self._predict_worker)
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
            fetch_worker = mps.Process(target=self._fetch_worker)
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
                if idx == DistillReader.POISON_PILL:
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
