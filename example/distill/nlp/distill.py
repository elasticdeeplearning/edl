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
from paddle_serving_client import Client
from paddle_serving_app.reader import ChineseBertReader

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--fixed_teacher",
    type=str,
    default=None,
    help="fixed teacher for debug local distill")
parser.add_argument("--s_weight", type=float, help="weight of student in loss")
parser.add_argument(
    "--LR", type=float, default=5e-5, help="weight of student in loss")
parser.add_argument(
    "--epoch_num", type=int, default=20, help="weight of student in loss")
parser.add_argument(
    "--T", type=float, default=None, help="weight of student in loss")
args = parser.parse_args()
print("args:", args)


def train_with_distill(train_reader,
                       test_reader,
                       word_dict,
                       orig_reader,
                       epoch_num=EPOCH):
    model = CNN(word_dict)
    g_clip = F.clip.GradientClipByGlobalNorm(1.0)  #experimental
    #opt = F.optimizer.Adam(learning_rate=LR, parameter_list=model.parameters(), grad_clip=g_clip)
    opt = AdamW(
        learning_rate=LR,
        parameter_list=model.parameters(),
        weight_decay=0.01,
        grad_clip=g_clip)
    model.train()
    for epoch in range(epoch_num):
        for step, output in enumerate(train_reader()):
            (_, _, _, _, ids_student, labels, logits_t) = output

            ids_student = D.base.to_variable(
                pad_batch_data(ids_student, 'int64'))
            labels = D.base.to_variable(np.array(labels).astype('int64'))
            logits_t = D.base.to_variable(np.array(logits_t).astype('float32'))

            _, logits_s = model(ids_student)  # student 模型输出logits
            loss_ce, _ = model(ids_student, labels=labels)
            if args.T is None:
                loss = args.s_weight / 100.0 * loss_ce + (
                    1.0 - args.s_weight / 100.0
                ) * L.softmax_with_cross_entropy(
                    logits_s, logits_t, soft_label=True)
            else:
                Tf = args.T / 10.0
                p = L.softmax(logits_t / Tf)
                loss = args.s_weight / 100.0 * loss_ce + (
                    1.0 - args.s_weight / 100.0
                ) * Tf * Tf * L.softmax_with_cross_entropy(
                    logits_s / Tf, p, soft_label=True)
            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] distill train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1, acc = evaluate_student(model, test_reader)
        print('teacher:with distillation student f1 %.5f acc %.5f' % (f1, acc))

        for step, (ids_student, labels, sentence) in enumerate(orig_reader()):
            loss, _ = model(ids_student, labels=labels)
            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()

        f1, acc = evaluate_student(model, test_reader)
        print('hard:with distillation student f1 %.5f acc %.5f' % (f1, acc))


def ernie_reader(s_reader, key_list):
    bert_reader = ChineseBertReader({
        'max_seq_len': 512,
        "vocab_file": "./data/vocab.txt"
    })

    def reader():
        for (ids_student, labels, ss) in s_reader():
            b = {}
            for k in key_list:
                b[k] = []

            #print("ids_student batch_size:", len(ids_student), len(labels), len(ss))
            for s in ss:
                feed_dict = bert_reader.process(s)
                for k in feed_dict:
                    #print("reader process:", k, len(feed_dict[k])
                    b[k].append(feed_dict[k])
            b["ids_student"] = ids_student
            b["labels"] = labels

            l = []
            for k in key_list:
                l.append(b[k])

            #print("ernie_reader batch_size:", len(l[0]))
            yield l

    return reader


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

    #train_without_distill(train_reader, dev_reader, word_dict)

    feed_keys = [
        "input_ids", "position_ids", "segment_ids", "input_mask",
        "ids_student", "labels"
    ]

    # distill reader and teacher
    dr = DistillReader(feed_keys, predicts=['logits'])
    dr.set_teacher_batch_size(batch_size)
    dr.set_serving_conf_file(
        "./ernie_senti_client/serving_client_conf.prototxt")
    if args.fixed_teacher:
        dr.set_fixed_teacher(args.fixed_teacher)

    dr_train_reader = ds.batch_reader(
        "./data/train.part.0", word_dict, batch_size=batch_size)
    dr_t = dr.set_batch_generator(ernie_reader(dr_train_reader, feed_keys))

    train_with_distill(dr_t, dev_reader, word_dict, train_reader)
