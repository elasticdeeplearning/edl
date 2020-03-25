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

import paddle
import paddle.fluid as fluid
import os
import sys
import numpy
import cPickle
import glob
from common import prepare_dataset, cluster_reader

OUTPUT_PATH = "./dataset/mnist/"
NAME_PREFIX = "mnist-train"

BATCH_SIZE = 128
USE_CUDA = False
BATCH_SIZE = 20
EPOCH_NUM = 100
NN_TYPE = "mlp"  # mlp or conv
params_dirname = "recognize_digits.inference.model"


def softmax_regression(img):
    predict = fluid.layers.fc(input=img, size=10, act='softmax')
    return predict


def multilayer_perceptron(img):
    # first fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=img, size=200, act='relu')
    # second fully-connected layer, using ReLu as its activation function
    hidden = fluid.layers.fc(input=hidden, size=200, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def convolutional_neural_network(img):
    # first conv pool
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    # second conv pool
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    # output layer with softmax activation function. size = 10 since there are only 10 possible digits.
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


def net_conf(img, label):
    #predict = softmax_regression(img)
    #predict = multilayer_perceptron(img)
    prediction = convolutional_neural_network(img)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def train():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    prediction, avg_loss, acc = net_conf(img, label)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    test_program = fluid.default_main_program().clone(for_test=True)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    def train_loop(train_prog, train_reader):
        for epoch in xrange(EPOCH_NUM):
            for batch_id, batch_data in enumerate(train_reader()):
                fluid.io.save_inference_model(
                    dirname=params_dirname,
                    feeded_var_names=["img"],
                    target_vars=[prediction],
                    executor=exe)
                exe.run(train_prog, feed=feeder.feed(batch_data))
                if batch_id > 0 and batch_id % 10 == 0:
                    acc_set = []
                    avg_loss_set = []
                    for test_data in test_reader():
                        acc_np, avg_loss_np = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=[acc, avg_loss])
                        acc_set.append(float(acc_np))
                        avg_loss_set.append(float(avg_loss_np))
                    # get test acc and loss
                    acc_val = numpy.array(acc_set).mean()
                    avg_loss_val = numpy.array(avg_loss_set).mean()

                    print("Epoch: {0}, Batch: {1}, Test Loss: {2}, Acc: {3}".
                          format(epoch, batch_id,
                                 float(avg_loss_val), float(acc_val)))
        exe.close()

    training_role = os.getenv("PADDLE_TRAINING_ROLE", None)
    if not training_role:
        # local training
        print("launch local training...")
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        exe.run(fluid.default_startup_program())
        train_loop(fluid.default_main_program(), train_reader)
    else:
        print("launch distributed training:")
        # distributed training
        pserver_endpoints = os.getenv(
            "PADDLE_PSERVER_EPS")  # the pserver server endpoint list
        trainers = int(os.getenv("PADDLE_TRAINERS"))  # total trainer count
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID",
                                   "0"))  # current trainer id
        current_endpoint = os.getenv(
            "PADDLE_CURRENT_ENDPOINT")  # current pserver endpoint
        print("training role: {0}\nps endpoint list: {1}\n"
              "trainers: {2}\ntrainer_id: {3}\n"
              "current ps endpoint: {4}\n".format(
                  training_role, pserver_endpoints, trainers, trainer_id,
                  current_endpoint))
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id=trainer_id,
            program=fluid.default_main_program(),
            pservers=pserver_endpoints,
            trainers=trainers)

        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            startup_prog = t.get_startup_program(current_endpoint,
                                                 pserver_prog)
            exe.run(startup_prog)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            cluster_train_reader = paddle.batch(
                paddle.reader.shuffle(
                    cluster_reader('dataset/mnist/mnist-train*', trainers,
                                   trainer_id),
                    buf_size=500),
                BATCH_SIZE)
            trainer_prog = t.get_trainer_program()
            exe.run(fluid.default_startup_program())
            train_loop(trainer_prog, cluster_train_reader)


def infer():
    import os
    import numpy as np
    from PIL import Image

    def _load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.getcwd()
    img = _load_image(cur_dir + '/image/infer_3.png')

    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)

    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

    assert feed_target_names[0] == 'img'
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: img},
                      fetch_list=fetch_targets)
    lab = np.argsort(results)
    print('Inference result of img/infer_3.png is: ', lab[0][0][-1])


if __name__ == "__main__":
    usage = "python recognize_digits.py [prepare|train|infer]"
    if len(sys.argv) != 2:
        print(usage)
        exit(0)
    flag = sys.argv[1]
    if flag == 'prepare':
        prepare_dataset(
            OUTPUT_PATH,
            NAME_PREFIX,
            paddle.dataset.mnist.train(),
            sample_count=4096)
    elif flag == "train":
        train()
    elif flag == "infer":
        infer()
    else:
        print(usage)
