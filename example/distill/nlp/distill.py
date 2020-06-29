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
from model import CNN, AdamW, evaluate_student, KL, BOW, KL_T, model_factory

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--fixed_teacher",
    type=str,
    default=None,
    help="fixed teacher for debug local distill")
parser.add_argument(
    "--s_weight", type=float, default=0.5, help="weight of student in loss")
parser.add_argument(
    "--epoch_num", type=int, default=10, help="weight of student in loss")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.01,
    help="weight of student in loss")
parser.add_argument(
    "--opt", type=str, default="AdamW", help="weight of student in loss")
parser.add_argument("--train_range", type=int, default=10, help="train range")
parser.add_argument(
    "--use_data_au", type=int, default=1, help="use data augmentation")
parser.add_argument(
    "--T", type=float, default=None, help="weight of student in loss")
parser.add_argument(
    "--model", type=str, default="BOW", help="student model name")
args = parser.parse_args()
print("parsed args:", args)

g_max_dev_acc = []
g_max_test_acc = []


def train_with_distill(train_reader, dev_reader, word_dict, test_reader,
                       epoch_num):
    model = model_factory(args.model, word_dict)
    if args.opt == "Adam":
        opt = F.optimizer.Adam(
            learning_rate=model.lr(steps_per_epoch=2250),
            parameter_list=model.parameters(),
            regularization=F.regularizer.L2Decay(
                regularization_coeff=args.weight_decay))
    else:
        opt = AdamW(
            learning_rate=model.lr(steps_per_epoch=2250),
            parameter_list=model.parameters(),
            weight_decay=args.weight_decay)

    model.train()
    max_dev_acc = 0.0
    max_test_acc = 0.0
    for epoch in range(epoch_num):
        for step, output in enumerate(train_reader()):
            (_, _, _, _, ids_student, labels, logits_t) = output

            ids_student = D.base.to_variable(
                pad_batch_data(ids_student, 'int64'))
            labels = D.base.to_variable(np.array(labels).astype('int64'))
            logits_t = D.base.to_variable(np.array(logits_t).astype('float32'))
            logits_t.stop_gradient = True

            _, logits_s = model(ids_student)
            loss_ce, _ = model(ids_student, labels=labels)

            if args.T is None:
                loss_kd = KL(logits_s, logits_t)
                loss = args.s_weight * loss_ce + (1.0 - args.s_weight
                                                  ) * loss_kd
            else:
                loss_kd = KL_T(logits_s, logits_t, args.T)
                loss = args.T * args.T * (loss_ce + loss_kd)
                #loss_kd = KL(logits_s, logits_t)
                #loss = loss_ce +  loss_kd

            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 100 == 0:
                print("stduent logits:", logits_s)
                print("teatcher logits:", logits_t)
                print('[step %03d] distill train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1, acc = evaluate_student(model, dev_reader)
        print('student on dev f1 %.5f acc %.5f epoch_no %d' % (f1, acc, epoch))

        if max_dev_acc < acc:
            max_dev_acc = acc

        f1, acc = evaluate_student(model, test_reader)
        print('student on test f1 %.5f acc %.5f epoch_no %d' %
              (f1, acc, epoch))

        if max_test_acc < acc:
            max_test_acc = acc

    g_max_dev_acc.append(max_dev_acc)
    g_max_test_acc.append(max_test_acc)


def ernie_reader(s_reader, key_list):
    bert_reader = ChineseBertReader({
        'max_seq_len': 256,
        "vocab_file": "./data/vocab.txt"
    })

    def reader():
        for (ids_student, labels, ss) in s_reader():
            b = {}
            for k in key_list:
                b[k] = []

            for s in ss:
                feed_dict = bert_reader.process(s)
                for k in feed_dict:
                    b[k].append(feed_dict[k])
            b["ids_student"] = ids_student
            b["labels"] = labels

            l = []
            for k in key_list:
                l.append(b[k])

            yield l

    return reader


def train():
    ds = ChnSentiCorp()
    word_dict = ds.student_word_dict("./data/vocab.bow.txt")
    batch_size = 16

    input_files = []
    if args.use_data_au:
        for i in range(1, 5):
            input_files.append("./data/train-data-augmented/part.{}".format(i))
    else:
        input_files.append("./data/train.part.0")

    # student train and dev
    train_reader = ds.pad_batch_reader(
        input_files, word_dict, batch_size=batch_size)
    dev_reader = ds.pad_batch_reader(
        "./data/dev.part.0", word_dict, batch_size=batch_size)
    test_reader = ds.pad_batch_reader(
        "./data/test.part.0", word_dict, batch_size=batch_size)

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
        input_files, word_dict, batch_size=batch_size)
    dr_t = dr.set_batch_generator(ernie_reader(dr_train_reader, feed_keys))

    train_with_distill(
        dr_t, dev_reader, word_dict, test_reader, epoch_num=args.epoch_num)


if __name__ == "__main__":
    place = F.CUDAPlace(0)
    D.guard(place).__enter__()

    for i in range(args.train_range):
        train()

        arr = np.array(g_max_dev_acc)
        print("max_dev_acc:", arr, "average:", np.average(arr), "train_args:",
              args)

        arr = np.array(g_max_test_acc)
        print("max_test_acc:", arr, "average:", np.average(arr), "train_args:",
              args)
