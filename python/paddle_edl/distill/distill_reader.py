# -*- coding: utf-8 -*-
import logging
import multiprocessing as mps
import os
import threading
import time
import six

from . import distill_worker

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


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
        self._type = os.environ.get('PADDLE_DISTILL_BALANCE_TYPE', 'redis')

    def _connect(self):
        if self._type == 'etcd':
            from paddle_edl.distill.discovery_client import DiscoveryClient
        elif self._type == 'redis':
            from paddle_edl.distill.redis.client import Client as DiscoveryClient
        else:
            assert False, 'BALANCE_TYPE must be etcd or redis'
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
    def __init__(self, ins, predicts):
        self._feeds = ins
        self._fetchs = predicts
        self._serving_conf_file = './serving_conf/serving_client_conf.prototxt'

        self._teacher_batch_size = 1

        self._mode = None
        self._teachers = []
        self._require_num = 1

        self._discovery_servers = []
        self._service_name = None

        # reader worker args
        self._reader_worker = None
        self._reader = None
        self._reader_type = None
        self._reader_out_queue = None
        self._reader_stop_event = None
        self._reader_cond = None

        self._task_semaphore = None

        # predict worker args
        self._predict_server_queue = None
        self._predict_server_result_queue = None
        self._working_predict_count = None
        self._predict_out_queue = None
        self._predict_stop_events = None
        self._predict_lock = None
        self._predict_finished_task = None
        self._predict_cond = None
        # predict worker pool
        self._predict_manage_thread = None
        self._predict_manage_stop_event = None

        # fetch args
        self._fetch_stop_event = None

        # work processor
        self._is_predict_start = False
        self._is_reader_start = False

        self._is_args_init = False

    def _get_servers(self, first_in):
        global _service_discover
        if _service_discover is not None:
            return _service_discover.get_servers()

        # FIXME. The order of object destruction
        if not first_in:
            logger.debug('service discover must have been deconstructed')
            return None

        with _service_discover_lock:
            assert first_in is True, 'service discover must be create in first in'
            while _service_discover is None:
                if self._mode == 'fixed':
                    _service_discover = FixedServiceDiscover(self._teachers)
                elif self._mode == 'discover':
                    _service_discover = DynamicServiceDiscover(
                        self._discovery_servers, self._require_num,
                        self._service_name)
                else:
                    raise TypeError('mode must be fixed or discover')
        return _service_discover.get_servers()

    def _start_reader_worker(self):
        if not self._is_reader_start:
            reader_worker = mps.Process(
                target=distill_worker.reader_worker,
                args=(
                    self._reader,
                    self._reader_type,
                    self._teacher_batch_size,
                    self._reader_out_queue,
                    self._reader_stop_event,
                    self._task_semaphore,
                    self._reader_cond, ))
            reader_worker.daemon = True
            reader_worker.start()
            self._reader_worker = reader_worker
            self._is_reader_start = True
        else:
            with self._reader_cond:
                self._reader_cond.notify()

    def _start_predict_worker(self):
        process = []
        for i in range(self._require_num):
            worker = mps.Process(
                target=distill_worker.predict_worker,
                args=(
                    self._predict_server_queue,
                    self._predict_server_result_queue,
                    self._working_predict_count,
                    self._reader_out_queue,
                    self._predict_out_queue,
                    self._feeds,
                    self._fetchs,
                    self._serving_conf_file,
                    self._predict_stop_events,
                    self._predict_lock,
                    self._predict_finished_task,
                    self._predict_cond, ))
            worker.daemon = True
            worker.start()
            process.append(worker)
        return process

    def _start_predict_worker_pool(self):
        if not self._is_predict_start:
            # start predict worker pool
            process = self._start_predict_worker()
            self._predict_manage_stop_event = threading.Event()
            self._predict_manage_thread = threading.Thread(
                target=distill_worker.predict_manage_worker,
                args=(
                    process,
                    self._predict_server_queue,
                    self._predict_server_result_queue,
                    self._require_num,
                    self._predict_stop_events,
                    self._get_servers,
                    self._predict_manage_stop_event,
                    self._predict_cond, ))
            self._predict_manage_thread.daemon = True
            self._predict_manage_thread.start()

            self._is_predict_start = True
        else:
            # wake up predict worker pool
            with self._predict_cond:
                self._predict_cond.notify_all()

    def _init_args(self):
        if not self._is_args_init:
            self._init_conf_file_from_env()
            self._init_discovery_from_env()

            # reader
            self._reader_out_queue = mps.Queue()
            self._reader_stop_event = mps.Event()
            self._reader_cond = mps.Condition()
            self._task_semaphore = mps.Semaphore(2 * self._require_num + 2)

            # predict
            self._predict_server_queue = mps.Queue(self._require_num)
            self._predict_server_result_queue = mps.Queue(self._require_num)
            self._working_predict_count = mps.Value('i', 0, lock=False)
            self._predict_out_queue = mps.Queue()
            self._predict_stop_events = [
                mps.Event() for i in range(self._require_num)
            ]
            self._predict_lock = mps.Lock()
            self._predict_finished_task = mps.Value('i', 0, lock=False)
            self._predict_cond = mps.Condition()

            # fetch
            self._fetch_stop_event = mps.Event()

            self._is_args_init = True

    def _init_conf_file_from_env(self):
        if os.path.isfile(self._serving_conf_file):
            # If there is a file in the default path, or the
            # user has set a conf file, then use this file.
            return

        conf_file = os.getenv('PADDLE_DISTILL_CONF_FILE')
        assert conf_file is not None, \
            'For now, you must put paddle_serving conf_file to the default ' \
            'path={}, or use `set_serving_conf_file` interface to set the path ' \
            'of conf_file. In the future, we will cancel the settings of this file'.\
            format(self._serving_conf_file)

        assert os.path.isfile(conf_file), \
            'Get serving conf_file={} from PaddleCloud environment, ' \
            'but it is not a file, please contact PaddleCloud administrator ' \
            'to troubleshoot the problem.'.format(conf_file)

        self._serving_conf_file = conf_file

    def _init_discovery_from_env(self):
        # env have highest priority
        discovery_servers = os.environ.get('PADDLE_DISTILL_BALANCE_SERVER')
        if discovery_servers is not None:
            service_name = os.environ.get('PADDLE_DISTILL_SERVICE_NAME')
            assert service_name is not None

            self._mode = 'discover'
            self._discovery_servers = discovery_servers.split(',')
            self._service_name = service_name

            max_teacher = os.environ.get('PADDLE_DISTILL_MAX_TEACHER')
            if max_teacher is not None:
                self._require_num = int(max_teacher)

        assert self._mode is not None, \
            'Teacher is empty, you can use `set_fixed_teacher` to set fixed teacher, ' \
            'or use `set_dynamic_teacher` to set the service discovery to automatically ' \
            'obtain the teacher. Or set the paddlecloud environment variable and obtain ' \
            'the discovery service from the environment'

    def set_serving_conf_file(self, conf_file):
        assert os.path.isfile(conf_file), '{} is not file'.format(conf_file)
        self._serving_conf_file = conf_file

    def set_teacher_batch_size(self, teacher_batch_size=1):
        self._teacher_batch_size = teacher_batch_size

    def set_fixed_teacher(self, teachers):
        if type(teachers) in (list, tuple):
            self._teachers = teachers
        elif type(teachers) == str:
            self._teachers = teachers.split(',')
        else:
            raise TypeError('teachers must be list|tuple|str')

        self._mode = 'fixed'
        self._require_num = len(self._teachers)

    def set_dynamic_teacher(self,
                            discovery_servers,
                            teacher_service_name,
                            require_max_teacher=1):
        if type(discovery_servers) in (list, tuple):
            self._discovery_servers = discovery_servers
        elif type(discovery_servers) == str:
            self._discovery_servers = discovery_servers.split(',')
        else:
            raise TypeError('discovery_servers must be list|tuple|str')

        self._mode = 'discover'
        self._service_name = teacher_service_name
        self._require_num = require_max_teacher

    def set_require_max_teacher(self, require_max_teacher):
        if self._mode == 'fixed':
            return
        self._require_num = require_max_teacher

    def set_sample_generator(self, reader):
        assert self._reader is None, 'reader has already set'
        self._reader = reader
        self._reader_type = distill_worker.ReaderType.SAMPLE
        return self

    def set_sample_list_generator(self, reader):
        assert self._reader is None, 'reader has already set'
        self._reader = reader
        self._reader_type = distill_worker.ReaderType.SAMPLE_LIST
        return self

    def set_batch_generator(self, reader):
        assert self._reader is None, 'reader has already set'
        self._reader = reader
        self._reader_type = distill_worker.ReaderType.BATCH
        return self

    def print_config(self):
        print("------ DistillReader Configuration Arguments ------")
        if not self._is_args_init:
            print("DistillReader not start yet, some args may change.")
        print_config = {
            'ins': self._feeds,
            'predicts': self._fetchs,
            'serving_conf_file': self._serving_conf_file,
            'teacher_batch_size': self._teacher_batch_size,
            'distill_mode': self._mode,
            'teachers': self._teachers,
            'require_max_teacher': self._require_num,
            'discovery_servers': self._discovery_servers,
            'teacher_service_name': self._service_name,
            'reader_type': self._reader_type,
        }
        for config, value in six.iteritems(print_config):
            print("%s: %s" % (config, value))
        print("------------------------------------------------")

    def __call__(self):
        assert self._reader is not None, "must set reader before iter DistillReader"

        self._init_args()

        assert self._reader_out_queue.empty()
        assert self._predict_out_queue.empty()

        self._start_reader_worker()
        # NOTE. When using logging, if start_predict_worker_pool is before
        # start_reader_worker, logging maybe hang!!!
        # For specific reasons, please see https://bugs.python.org/issue6721
        # https://stackoverflow.com/questions/24509650/deadlock-with-logging-multiprocess-multithread-python-script
        # >>> The problem is common in any situation where you have locks, threads and forks.
        # >>> If thread 1 had a lock while thread 2 calls fork, in the forked process,
        # >>> there will only be thread 2 and the lock will be held forever.
        # So need to move start_predict_worker_pool to the end if we use logging in predict
        # manager thread, or for the sake of safety, don't use logging?
        self._start_predict_worker_pool()

        for data in distill_worker.fetch_out(
                self._reader_type, self._predict_out_queue,
                self._fetch_stop_event, self._task_semaphore):
            yield data

    def __del__(self):
        if not self._is_args_init:
            return

        # stop reader worker
        with self._reader_cond:
            self._reader_stop_event.set()
            self._reader_cond.notify()

        self._predict_manage_stop_event.set()

        for i in range(20):
            if self._reader_worker.is_alive() or \
               self._predict_manage_thread.is_alive():
                time.sleep(1)
