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

import sys

from edl.utils import common_pb2


class EdlException(Exception):
    pass


class EdlStopIteration(EdlException):
    pass


class EdlRegisterError(EdlException):
    pass


class EdlBarrierError(EdlException):
    pass


class EdlUnkownError(EdlException):
    pass


class EdlRankError(EdlException):
    pass


class EdlInternalError(EdlException):
    pass


class EdlWaitFollowersReleaseError(EdlException):
    pass


class EdlLeaderError(EdlException):
    pass


class EdlGenerateClusterError(EdlException):
    pass


class EdlTableError(EdlException):
    pass


class EdlEtcdIOError(EdlException):
    pass


class EdlDataEndError(EdlException):
    pass


class EdlPodIDNotExistError(EdlException):
    pass


class EdlReaderNameError(EdlException):
    pass


class EdlFileListNotMatchError(EdlException):
    pass


class EdlDataGenerateError(EdlException):
    pass


class EdlNotLeaderError(EdlException):
    pass


class EdlNotFoundLeader(EdlException):
    pass


class EdlAccessDataError(EdlException):
    pass


def deserialize(pb_status):
    thismodule = sys.modules[__name__]
    try:
        cls = getattr(thismodule, pb_status.type)(pb_status.detail)
    except Exception as e:
        raise Exception(
            "type:{} detail:{} meets error:{}".format(
                pb_status.type, pb_status.detail, str(e)
            )
        )
    raise cls


def serialize_to_pb_status(exception):
    pb_status = common_pb2.Status()
    pb_status.type = exception.__class__.__name__
    pb_status.detail = str(exception)
    return pb_status


def serialize(pb_response, exception, stack_info=None):
    pb_response.status.type = exception.__class__.__name__
    if stack_info is not None:
        pb_response.status.detail = str(exception) + stack_info
    else:
        pb_response.status.detail = str(exception)
