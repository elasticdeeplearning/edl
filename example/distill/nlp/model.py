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
