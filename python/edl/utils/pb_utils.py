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

import json


def record_to_string(rec):
    return "record_no:{} fields_len:{}".format(rec.record_no,
                                               len(rec.field_data))


def batch_data_response_to_string(res):
    r = []
    for data in res.data:
        s = {}
        s["batch_data_id"] = data.batch_data_id
        s["records_num"] = len(data.records)

        records_str = []
        for rec in data.records:
            records_str.append(record_to_string(rec))

        s["records"] = ",".join(records_str)
        r.append(json.dumps(s))

    return ";".jion(r)


def batch_data_meta_response_to_string(res):
    r = []
    for data in res.data:
        r.append(str(data))

    return ",".join(r)
