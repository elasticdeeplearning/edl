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
    "--T", type=float, default=None, help="weight of student in loss")
args = parser.parse_args()
print("args:", args)

EPOCH = 10
LR = 5e-5


class AdamW(F.optimizer.AdamOptimizer):
    """AdamW object for dygraph"""

    def __init__(self, *args, **kwargs):
        weight_decay = kwargs.pop('weight_decay', None)
        var_name_to_exclude = kwargs.pop(
            'var_name_to_exclude',
            '.*layer_norm_scale|.*layer_norm_bias|.*b_0')
        super(AdamW, self).__init__(*args, **kwargs)
        self.wd = weight_decay
        self.pat = re.compile(var_name_to_exclude)

    def apply_optimize(self, loss, startup_program, params_grads):
        super(AdamW, self).apply_optimize(loss, startup_program, params_grads)
        for p, g in params_grads:
            #log.debug(L.reduce_mean(p))
            if not self.pat.match(p.name):
                L.assign(p * (1. - self.wd * self.current_step_lr()), p)
            #log.debug(L.reduce_mean(p))


def KL(pred, target):
    #print("KL", pred.shape, target.shape)
    pred = L.log(L.softmax(pred))
    target = L.softmax(target)
    loss = L.kldiv_loss(pred, target)
    return loss


def evaluate_student(model, test_reader):
    all_pred, all_label = [], []
    with D.base._switch_tracer_mode_guard_(is_train=False):
        model.eval()
        for step, (ids_student, labels, _) in enumerate(test_reader()):
            _, logits = model(ids_student)
            pred = L.argmax(logits, -1)
            all_pred.extend(pred.numpy())
            all_label.extend(labels.numpy())
        f1 = f1_score(all_label, all_pred, average='macro')
        acc = accuracy_score(all_label, all_pred)
        model.train()
        return f1, acc


class BOW(D.Layer):
    def __init__(self, word_dict):
        super().__init__()
        self.emb = D.Embedding([len(word_dict), 128], padding_idx=0)
        self.fc = D.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        pad_mask = L.unsqueeze(L.cast(ids != 0, 'float32'), [-1])

        embbed = L.reduce_sum(embbed * pad_mask, 1)
        embbed = L.softsign(embbed)
        logits = self.fc(embbed)
        #print("embbed:", embbed.shape,
        #      "pad_mask", pad_mak.shape)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


class CNN(D.Layer):
    def __init__(self, word_dict):
        super().__init__()
        self.emb = D.Embedding([len(word_dict), 128], padding_idx=0)
        self.cnn = D.Conv2D(128, 128, (1, 3), padding=(0, 1), act='relu')
        self.pool = D.Pool2D((1, 3), pool_padding=(0, 1))
        self.fc = D.Linear(128, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        #print("ids shape:", ids.shape)
        #d_batch, d_seqlen = ids.shape
        hidden = embbed
        hidden = L.transpose(hidden, [0, 2, 1])  #change to NCWH
        hidden = L.unsqueeze(hidden, [2])
        hidden = self.cnn(hidden)
        hidden = self.pool(hidden)
        hidden = L.squeeze(hidden, [2])
        hidden = L.transpose(hidden, [0, 2, 1])
        pad_mask = L.unsqueeze(L.cast(ids != 0, 'float32'), [-1])
        hidden = L.softsign(L.reduce_sum(hidden * pad_mask, 1))
        logits = self.fc(hidden)
        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
            #loss = L.reduce_mean(loss)
        else:
            loss = None
        return loss, logits


def train_without_distill(train_reader,
                          test_reader,
                          word_dict,
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
        for step, (ids_student, labels, sentence) in enumerate(train_reader()):
            #print(ids_student.shape, labels.shape,  sentence)
            #sys.exit(0)
            loss, _ = model(ids_student, labels=labels)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] distill train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        #f1 = evaluate_student(model, test_reader)
        #print('without distillation student f1 %.5f' % f1)
        f1, acc = evaluate_student(model, test_reader)
        print('without distillation student f1 %.5f acc %.5f' % (f1, acc))


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

            #pooled_output = D.base.to_variable(np.array(pooled_output)).astype('float32')
            #print(pooled_output.shape, logits_t.shape)

            ids_student = D.base.to_variable(
                pad_batch_data(ids_student, 'int64'))
            labels = D.base.to_variable(np.array(labels).astype('int64'))
            logits_t = D.base.to_variable(np.array(logits_t).astype('float32'))

            _, logits_s = model(ids_student)  # student 模型输出logits
            loss_ce, _ = model(ids_student, labels=labels)
            #loss_kd = KL(logits_s, logits_t)    # 由KL divergence度量两个分布的距离
            #loss =  loss_ce +  loss_kd
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
            #loss = L.softmax_with_cross_entropy(logits_s, logits_t, soft_label=True)
            loss = L.reduce_mean(loss)
            loss.backward()
            if step % 10 == 0:
                print('[step %03d] distill train loss %.5f lr %.3e' %
                      (step, loss.numpy(), opt.current_step_lr()))
            opt.minimize(loss)
            model.clear_gradients()
        f1, acc = evaluate_student(model, test_reader)
        print('teacher:with distillation student f1 %.5f acc %.5f' % (f1, acc))

        # 最后再加一轮hard label训练巩固结果
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
