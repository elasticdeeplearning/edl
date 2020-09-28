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

# https://stackoverflow.com/questions/6879501/filter-git-diff-by-type-of-change
changed_sh_files=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.sh$')
if [[ "${changed_sh_files}" == "" ]]; then
    exit 0
fi

echo "${changed_sh_files}" | xargs shellcheck
