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

from model import CNN, AdamW, evaluate_student

g_max_acc = []


def train_without_distill(train_reader, test_reader, word_dict, epoch_num, lr):
    model = CNN(word_dict)
    g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
    #opt = F.optimizer.Adam(learning_rate=lr, parameter_list=model.parameters(), grad_clip=g_clip)
    opt = AdamW(
        learning_rate=lr,
        parameter_list=model.parameters(),
        weight_decay=0.01,
        grad_clip=g_clip)
    model.train()

    max_acc = 0.0
    for epoch in range(epoch_num):
        for step, (ids_student, labels, sentence) in enumerate(train_reader()):
            loss, _ = model(ids_student, labels=labels)
            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] distill train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1, acc = evaluate_student(model, test_reader)
        print('without distillation student f1 %.5f acc %.5f' % (f1, acc))

        if max_acc < acc:
            max_acc = acc

    g_max_acc.append(max_acc)


if __name__ == "__main__":
    place = F.CUDAPlace(0)
    D.guard(place).__enter__()

    ds = ChnSentiCorp()
    word_dict = ds.student_word_dict("./data/vocab.bow.txt")
    batch_size = 16

    # student train and dev
    train_reader = ds.pad_batch_reader(
        "./data/train.part.0", word_dict, batch_size=batch_size)
    dev_reader = ds.pad_batch_reader(
        "./data/dev.part.0", word_dict, batch_size=batch_size)

    for i in range(10):
        train_without_distill(
            train_reader, dev_reader, word_dict, epoch_num=10, lr=5e-5)

    arr = np.array(g_max_acc)
    print("max_acc:", arr, "average:", np.average(arr))
