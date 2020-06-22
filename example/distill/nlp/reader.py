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

import codecs
import os
import csv
import sys

from paddlehub.dataset import InputExample
from paddlehub.common.dir import DATA_HOME
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
import paddle as P
import paddle.fluid.dygraph as D
import numpy as np


def space_tokenizer(i):
    return i.split()


def pad_batch_data(data, dtype, pad_idx=0, max_len=-1):
    if max_len <= 0:
        for s in data:
            if len(s) > max_len:
                max_len = len(s)

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in data
    ])

    return np.array(inst_data).astype(dtype)


class ChnSentiCorp(BaseNLPDataset):
    def __init__(self):
        base_path = "./data/"
        super(ChnSentiCorp, self).__init__(
            base_path=base_path,
            train_file="train.part.0",
            dev_file="dev.part.0",
            test_file="test.part.0",
            label_file=None,
            label_list=["0", "1"], )

        self._word_dict = None

    def __read_file(self, input_file):
        """
        data file format:
        origin sentence\tword segment sentence\tlabel
        """
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                arr = line.split("\t")
                #print("line:", len(arr))
                yield arr

    def _read_file(self, input_file, phase=None):
        """
        [(seq_id,label,origin sentence)]
        """
        seq_id = 0
        examples = []
        for t in self.__read_file(input_file):
            if len(t) == 2:
                #example = InputExample(
                #    guid=seq_id, label=t[1], text_a=t[0])
                #print("t2", t[1])
                assert len(t) != 2, "data format error:" + t
            elif len(t) == 3:
                example = InputExample(guid=seq_id, label=t[2], text_a=t[0])
                #print("t3", t[2])
            else:
                assert False, 'invalid format'
            seq_id += 1
            examples.append(example)

        return examples

    def student_word_dict(self, vocab_file):
        """
        {
            word->word_idx
        }
        """
        with codecs.open(vocab_file, "r", encoding="UTF-8") as f:
            self._word_dict = {
                i.strip(): l
                for l, i in enumerate(f.readlines())
            }

        return self._word_dict

    def student_reader(self, input_files, word_dict):
        """
        return [([segment_sentence_idxs], label, sentence), ()...]
        """

        def reader():
            for data_file in input_files:
                print("open file:", data_file)
                for t in self.__read_file(data_file):
                    s = []
                    for word in space_tokenizer(t[1]):
                        idx = word_dict[
                            word] if word in word_dict else word_dict['[UNK]']
                        s.append(idx)

                    yield s, t[2], t[0]

        return reader

    def batch_reader(self, input_file, word_dict, batch_size):
        def reader():
            s_reader = P.reader.shuffle(
                self.student_reader(input_file, word_dict), buf_size=2000)

            b = [[], [], []]
            for rec in s_reader():
                if len(b[0]) == batch_size:
                    yield b
                    b = [[], [], []]
                    continue

                for i in range(len(rec)):
                    b[i].append(rec[i])

            if len(b[0]) > 0:
                yield b

        return reader

    def pad_batch_reader(self, input_file, word_dict, batch_size):
        def reader():
            b_reader = self.batch_reader(input_file, word_dict, batch_size)
            for b in b_reader():
                b[0] = D.base.to_variable(pad_batch_data(b[0], 'int64'))
                b[1] = D.base.to_variable(np.array(b[1]).astype('int64'))
                yield b

        return reader


if __name__ == '__main__':
    ds = ChnSentiCorp()
    ds._read_file("./data/train.part.0")
    ds.student_reader("./data/train.part.0", "./data/vocab.bow.txt")
