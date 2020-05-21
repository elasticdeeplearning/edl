# -*- coding: utf-8 -*-
import ast
import ctypes
import logging
import multiprocessing as mps
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")


class Var(object):
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


class DataStruct(object):
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
                self._all_vars.append(Var(_var, var_type, var_shape))

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


class SharedMemoryArray(object):
    def __init__(self, config, d_batch_size, batch_size, shared_len, capacity):
        self._d_batch_size = d_batch_size
        self._batch_size = batch_size
        self._data_struct = DataStruct(config, d_batch_size, batch_size)

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
