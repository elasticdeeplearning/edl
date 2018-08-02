#!/bin/env python
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.dataset as dataset
import numpy as np
import cPickle
import glob
import os
import sys
import math

OUTPUT_PATH="./dataset/mnist/"
NAME_PREFIX="mnist-train"
LINE_PER_FILE=1000
N = 5
EMBED_SIZE = 32
HIDDEN_SIZE = 256
BATCH_SIZE = 32
IS_SPARSE = False
ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy
def prepare_dataset(line_count=10000):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    word_dict = dataset.imikolov.build_dict()
    suffix = "%s/%s-%%05d.pickle" % (OUTPUT_PATH, NAME_PREFIX)
    lines = []
    indx_f = 0
    for i, d in enumerate(dataset.imikolov.train(word_dict, N)()):
        lines.append(d)
        if i >= line_count and i % line_count == 0:
            with open(suffix % indx_f, "w") as f:
                cPickle.dump(lines, f)
                lines = []
                indx_f += 1
    if lines:
        with open(suffix % indx_f, "w") as f:
            cPickle.dump(lines, f)

def cluster_reader(trainers, trainer_id):
    def reader():
        flist = glob.glob(os.path.join(OUTPUT_PATH, NAME_PREFIX + "*"))
        flist.sort()
        my_file_list = []
        for idx, fn in enumerate(flist):
            if idx % trainers == trainer_id:
                print("append file for current trainer: %s" % fn)
                my_file_list.append(fn)

        for fn in my_file_list:
            print("processing file: ", fn)
            with open(fn, "r") as f:
                lines = cPickle.load(f)
                for line in lines:
                    yield line
    return reader

def main():
    def __network__(words):
        embed_first = fluid.layers.embedding(
            input=words[0],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_second = fluid.layers.embedding(
            input=words[1],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_third = fluid.layers.embedding(
            input=words[2],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_forth = fluid.layers.embedding(
            input=words[3],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')

        concat_embed = fluid.layers.concat(
            input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
        hidden1 = fluid.layers.fc(input=concat_embed,
                                  size=HIDDEN_SIZE,
                                  act='sigmoid')
        predict_word = fluid.layers.fc(input=hidden1,
                                       size=dict_size,
                                       act='softmax')
        cost = fluid.layers.cross_entropy(input=predict_word, label=words[4])
        avg_cost = fluid.layers.mean(cost)
        return avg_cost, predict_word

    word_dict = dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    avg_cost, predict_word = __network__(
        [first_word, second_word, third_word, forth_word, next_word])

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    # Distribute Environment
    pserver_endpoints = os.getenv("PADDLE_PSERVER_EPS")     # the pserver server endpoint list
    trainers = int(os.getenv("PADDLE_TRAINERS"))            # total trainer count
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))   # current trainer id
    training_role = os.getenv("PADDLE_TRAINING_ROLE")       # current training role, should be in [trainer,pserver] 
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT") # current pserver endpoint

    train_reader = paddle.batch(cluster_reader(trainers, trainer_id), BATCH_SIZE)

    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id = trainer_id,                   
        program = fluid.default_main_program(),    
        pservers = pserver_endpoints,             
        trainers = trainers)                       
    if training_role == "pserver":
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    elif training_role == "trainer":
        trainer_prog = t.get_trainer_program()
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        use_gpu = True if core.is_compiled_with_cuda() else False

        exec_strategy = ExecutionStrategy()
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_gpu,
            main_program=trainer_prog,
            loss_name=avg_cost.name,
            exec_strategy=exec_strategy)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.itervalues()
            if var.is_data
        ]
        feeder = fluid.DataFeeder(feed_var_list, place)
        for pass_id in xrange(10):
            for batch_id, data in enumerate(train_reader()):
                avg_loss_np = train_exe.run(feed=feeder.feed(data),
                                            fetch_list=[avg_cost.name])
                loss = np.array(avg_loss_np).mean()
                print("Pass: {0}, Batch: {1}, Loss: {2}".format(pass_id, batch_id, loss))
                if float(loss) < 5.0:
                    exe.close()
                    return
                if math.isnan(loss):
                    assert ("Got Nan loss, training failed")
        
    else:
        raise AssertionError("PADDLE_TRAINING_ROLE should be in [PSERVER, TRAINER]")


if __name__ == "__main__":
    usage = "python dist_word2vec.py [prepare|train]"
    if len(sys.argv) != 2:
        print(usage)

    act = sys.argv[1]
    if act == 'prepare':
        prepare_dataset()
    elif act == "train":
        main()
    else:
        print(usage)
