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


class EdlExeception(Exception):
    pass


class EdLDuplicateInitDataSetError(EdlExeception):
    pass


class EdlDataSetEndError(EdlExeception):
    pass


class EdlRegisterError(EdlExeception):
    pass


class EdlBarrierError(EdlExeception):
    pass


class EdlUnkownError(EdlExeception):
    pass


"""
_excpetions = {
    "DuplicateInitDataSetError": EdlDuplicateInitDataSetError,
    "BarrierError": EdlBarrierError,
}


def edl_exception(e_type, detail):
    if e_type not in _excpetions:
        return EdlUnkownError(detail)

    e = _excpetions["e_type"](detail)
    raise e
"""
