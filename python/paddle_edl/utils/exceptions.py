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


def raise_exeception(name, detail):
    thismodule = sys.modules[__name__]
    cls = getattr(thismodule, name)(detail)
    raise cls


def get_instance_name(instance):
    return instance.__class__.__name__
