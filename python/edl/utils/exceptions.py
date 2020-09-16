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

from ..utils import common_pb2 as common_pb


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


def deserialize_exception(s):
    thismodule = sys.modules[__name__]
    try:
        cls = getattr(thismodule, s.type)(s.detail)
    except Exception as e:
        raise Exception("type:{} detail:{}".format(s.type, s.detail))
    #print(type(cls))
    raise cls


def serialize_exception(e):
    s = common_pb.Status()
    s.type = e.__class__.__name__
    s.detail = str(e)
    return s


def serialize_exception(res, e, stack_info=None):
    res.status.type = e.__class__.__name__
    if stack_info is not None:
        res.status.detail = str(e) + stack_info
    else:
        res.status.detail = str(e)
