#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Finetuning on classification task """

import argparse
import ast
import paddle.fluid as fluid
import paddlehub as hub
from reader import ChnSentiCorp
from paddle.fluid.transpiler.details import program_to_code
import paddle_serving_client.io as serving_io
from paddlehub.reader.nlp_reader import ClassifyReader
#from paddle.fluid.transpiler.details import program_to_code

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=5, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=256, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

class TextClassifierTask(hub.TextClassifierTask):
    def __init__(self,
                 num_classes,
                 feed_list,
                 data_reader,
                 feature=None,
                 token_feature=None,
                 network=None,
                 startup_program=None,
                 config=None,
                 hidden_units=None,
                 metrics_choices="default"):
        super(TextClassifierTask, self).__init__(num_classes=num_classes,
                                                 feed_list=feed_list,
                                                 data_reader=data_reader,
                                                 feature=feature,
                                                 token_feature=token_feature,
                                                 network=network,
                                                 startup_program=startup_program,
                                                 config=config,
                                                 hidden_units=hidden_units,
                                                 metrics_choices=metrics_choices)
        self._logits = None

    def _build_net(self):
        if isinstance(self._base_data_reader, ClassifyReader):
            # ClassifyReader will return the seqence length of an input text
            self.seq_len = fluid.layers.data(
                name="seq_len", shape=[1], dtype='int64', lod_level=0)
            self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])

            # unpad the token_feature
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len_used)

        if self.network:
            # add pre-defined net
            net_func = getattr(net.classification, self.network)
            if self.network == 'dpcnn':
                # deepcnn network is no need to unpad
                cls_feats = net_func(
                    self.feature, emb_dim=self.feature.shape[-1])
            else:
                cls_feats = net_func(unpad_feature)
            logger.info(
                "%s has been added in the TextClassifierTask!" % self.network)
        else:
            # not use pre-defined net but to use fc net
            cls_feats = fluid.layers.dropout(
                x=self.feature,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")

        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        self._logits = fluid.layers.fc(
            input=cls_feats,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))


        logits = fluid.layers.softmax(self._logits)

        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(logits, axis=1), shape=[-1, 1])

        return [logits]

def save_model(inputs, output, program, logits):
    feed_keys = ["input_ids", "position_ids", "segment_ids", "input_mask"]
    feed_dict = dict(zip(feed_keys, [inputs[x] for x in feed_keys]))
    fetch_keys = ["pooled_output", "sequence_output"]
    fetch_dict = dict(zip(fetch_keys, [outputs[x] for x in fetch_keys]))
    fetch_dict["logits"]=logits

    serving_io.save_model("ernie_senti_server", "ernie_senti_client", feed_dict, fetch_dict, main_program=program)

if __name__ == '__main__':

    # Load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use accuracy as metrics
    # metric should be acc, f1 or matthews
    #dataset = hub.dataset.ChnSentiCorp()
    dataset = ChnSentiCorp()
    metrics_choices = ["f1", "acc"]

    # For ernie_tiny, it use sub-word to tokenize chinese sentence
    # If not ernie tiny, sp_model_path and word_dict_path should be set None
    #print("vocab_file:", module.get_vocab_path())
    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a classfication finetune task by PaddleHub's API
    cls_task = TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=metrics_choices)

    logits=None
    program = None
    with cls_task.phase_guard('train'):
        for l in cls_task.outputs:
            print("cls_task outputs:", l)
        logits = cls_task._logits
        print("logits:", logits)
        program = cls_task.main_program

    if args.checkpoint_dir:
        cls_task.load_checkpoint()

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    cls_task.finetune_and_eval()
    save_model(inputs, outputs, program, logits)
