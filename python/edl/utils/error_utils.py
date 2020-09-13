import functools
import time

from . import exceptions
from .log_utils import logger

def handle_errors_until_timeout(f):
    def handler(*args, **kwargs):
        begin = time.time()
        timeout = kwargs['timeout']
        while True:
            try:
                return f(*args, **kwargs)
            except exceptions.EdlDataEndError as e:
                raise exceptions.EdlDataEndError
            except Exception as e:
                if time.time() - begin >= timeout:
                    logger.warning("{} execute timeout:{}".format(f.__name__))
                    raise e

                time.sleep(3)
                continue

    return functools.wraps(f)(handler)