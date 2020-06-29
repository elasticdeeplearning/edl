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
from text_basic import LSTM as basic_lstm


class LSTM(D.Layer):
    def __init__(self, word_dict):
        super().__init__()

        self.emb = D.Embedding([len(word_dict), 300])
        self.lstm = basic_lstm(input_size=300, hidden_size=150)
        self.fc = D.Linear(150, 2)

    def forward(self, ids, labels=None):
        embbed = self.emb(ids)
        #print("embed shape:", embbed.shape)

        lstm_out, hidden = self.lstm(embbed)
        #print("lstm_out shape:", lstm_out.shape)
        #print("hiden list len:", len(hidden))

        logits = self.fc(lstm_out[:, -1])
        #print("logits shape:", logits.shape)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = L.reshape(labels, [-1, 1])
            loss = L.softmax_with_cross_entropy(logits, labels)
        else:
            loss = None

        return loss, logits

    def lr(self, steps_per_epoch=None):
        return 1e-3
