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
import os

import numpy as np
import argparse
from sklearn.metrics import f1_score, accuracy_score
import paddle as P
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
from reader import ChnSentiCorp, pad_batch_data
from paddle_edl.distill.distill_reader import DistillReader
import re

import os
import sys

from model import CNN, AdamW, evaluate_student, BOW, model_factory

g_max_dev_acc = []
g_max_test_acc = []

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--model", type=str, default="BOW", help="student model name")
parser.add_argument(
    "--epoch_num", type=int, default=10, help="weight of student in loss")
parser.add_argument("--train_range", type=int, default=10, help="train range")
args = parser.parse_args()
print("parsed args:", args)


def train_without_distill(train_reader, dev_reader, test_reader, word_dict,
                          epoch_num):
    model = model_factory(args.model, word_dict)
    opt = AdamW(
        learning_rate=model.lr(),
        parameter_list=model.parameters(),
        weight_decay=0.01)
    model.train()

    max_dev_acc = 0.0
    max_test_acc = 0.0
    for epoch in range(epoch_num):
        for step, (ids_student, labels, sentence) in enumerate(train_reader()):
            _, logits_s = model(ids_student)
            loss, _ = model(ids_student, labels=labels)
            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 100 == 0:
                print('[step %03d] train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1, acc = evaluate_student(model, dev_reader)
        print('train_without_distill on dev f1 %.5f acc %.5f epoch_no %d' %
              (f1, acc, epoch))

        if max_dev_acc < acc:
            max_dev_acc = acc

        f1, acc = evaluate_student(model, test_reader)
        print('train_without_distill on test f1 %.5f acc %.5f epoch_no %d' %
              (f1, acc, epoch))

        if max_test_acc < acc:
            max_test_acc = acc

    g_max_dev_acc.append(max_dev_acc)
    g_max_test_acc.append(max_test_acc)


def train():
    ds = ChnSentiCorp()
    word_dict = ds.student_word_dict("./data/vocab.bow.txt")
    batch_size = 16

    # student train and dev
    train_reader = ds.pad_batch_reader(
        "./data/train.part.0", word_dict, batch_size=batch_size)
    dev_reader = ds.pad_batch_reader(
        "./data/dev.part.0", word_dict, batch_size=batch_size)
    test_reader = ds.pad_batch_reader(
        "./data/test.part.0", word_dict, batch_size=batch_size)

    train_without_distill(
        train_reader,
        dev_reader,
        test_reader,
        word_dict,
        epoch_num=args.epoch_num)


if __name__ == "__main__":
    place = F.CUDAPlace(0)
    D.guard(place).__enter__()

    for i in range(args.train_range):
        train()

        arr = np.array(g_max_dev_acc)
        print("max_dev_acc:", arr, "average:", np.average(arr))

        arr = np.array(g_max_test_acc)
        print("max_test_acc:", arr, "average:", np.average(arr))
