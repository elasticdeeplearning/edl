import paddle
import paddle.fluid as fluid
import os
import sys
import numpy
from common import prepare_dataset, cluster_reader

OUTPUT_PATH="./dataset/fit_a_line/"
NAME_PREFIX="fit_a_line-train"

USE_CUDA = False
BATCH_SIZE = 20
EPOCH_NUM = 100
params_dirname = "fit_a_line.inference.model"

def train():
    # add forward pass
    x = fluid.layers.data(name="x", shape=[13], dtype="float32")

    y = fluid.layers.data(name="y", shape=[1], dtype="float32")

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    # add backward pass and optimize pass
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    USE_CUDA = False
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()

    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

    exe = fluid.Executor(place)

    def train_loop(trainer_prog, train_reader):
        for epoch in range(EPOCH_NUM):
            fluid.io.save_inference_model(
                dirname = params_dirname,
                feeded_var_names = ["x"],
                target_vars = [y_predict],
                executor = exe)

            for batch_id, batch_data in enumerate(train_reader()):
                avg_loss_value, = exe.run(trainer_prog,
                                          feed=feeder.feed(batch_data),
                                          fetch_list=[avg_cost])
                if batch_id % 10 == 0 and batch_id > 0:
                    print("Epoch: {0}, Batch: {1}, loss: {2}".format(
                        epoch, batch_id, avg_loss_value[0]))
        exe.close()

    training_role = os.getenv("PADDLE_TRAINING_ROLE", None)
    if not training_role:
        # local training
        print("launch local training...")
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.uci_housing.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        exe.run(fluid.default_startup_program())
        train_loop(fluid.default_main_program(), train_reader)
    else:
        print("launch distributed training:")
        # distributed training
        pserver_endpoints = os.getenv("PADDLE_PSERVER_EPS")     # the pserver server endpoint list
        trainers = int(os.getenv("PADDLE_TRAINERS"))            # total trainer count
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))   # current trainer id
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT") # current pserver endpoint
        print("training role: {0}\nps endpoint list: {1}\n"
              "trainers: {2}\ntrainer_id: {3}\n"
              "current ps endpoint: {4}\n".format(
                  training_role, pserver_endpoints, trainers, trainer_id, current_endpoint))
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id = trainer_id,                   
            program = fluid.default_main_program(),    
            pservers = pserver_endpoints,             
            trainers = trainers)          

        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(startup_prog)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            cluster_train_reader = paddle.batch(
                paddle.reader.shuffle(cluster_reader(trainers, trainer_id), buf_size=500), BATCH_SIZE)
            trainer_prog = t.get_trainer_program()
            exe.run(fluid.default_startup_program())
            train_loop(trainer_prog, cluster_train_reader)

def infer():
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    exe = fluid.Executor(place)
    batch_size = 10
    [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

    test_reader = paddle.batch(
        paddle.dataset.uci_housing.test(), batch_size=batch_size)

    test_data = test_reader().next()
    test_feat = numpy.array(
        [data[0] for data in test_data]).astype("float32")
    test_label = numpy.array(
        [data[1] for data in test_data]).astype("float32")

    assert feed_target_names[0] == 'x'
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: numpy.array(test_feat)},
                      fetch_list=fetch_targets)
    print("infer shape: ", results[0].shape)
    print("infer results: ", results[0])
    print("ground truth: ", test_label)

if __name__ == "__main__":
    usage = "python fit_a_line.py [prepare|train|infer]"
    if len(sys.argv) != 2:
        print(usage)
        exit(0)
    act = sys.argv[1]
    if act == 'prepare':
        prepare_dataset(OUTPUT_PATH, NAME_PREFIX, paddle.dataset.uci_housing.train(), sample_count=128)
    elif act == "train":
        train()
    elif act == "infer":
        infer()
    else:
        print(usage)
