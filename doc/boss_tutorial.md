# PaddlePaddle Elastic Deep Learning

<img src="../logo/edl.png" width="500">

PaddlePaddle Elastic Deep Learning (EDL) is a clustering project which leverages Deep Learning training jobs to
be scalable and fault-tolerant. EDL will greatly boost the parallel distributed training jobs and make good use
of cluster computing power.

EDL is based on the full fault-tolerant feature of PaddlePaddle, it uses a Kubernetes controller to manage
the cluster training jobs and an auto-scaler to scale the job's computing resources.

For researchers, EDL with Kuberntes will reduce the waiting time of the job submitted, to help with
exposing potential algorithmic problems as early as possible.

For enterprises, industrial users tend to run deep learning jobs as a subset of the complete data pipeline,
including web servers and log collectors. EDL make it possible to run less deep learning job processes during
periods of high web traffic, more when web traffic is low. EDL would optimize the global utilization
of a cluster.

## Tutorial Outline

- Introduction
  At the introduction session, we will introduce:
    1. a new version named Fluid on PaddlePaddle; and
    1. Why we develop PaddlePaddle EDL and how to implement it.
- Hands-on tutorial
  Following the introduction, we will prepare a hands-on tutorial so that all the audience can use
  PaddlePaddle and ask some questions while using PaddlePaddle:
  - Part-1, Train a simple model using PaddlePaddle Fluid.
  - Part-2, Launch an EDL training job on a Kubernetes Cluster.

## Prerequisites

- [Install Docker](https://docs.docker.com/install/)
- [Install kubectl](./install.md#kubectl)
- A Kubernetes cluster which version is `1.7.x`
  - [minikube would launch a kubernetes cluster locally](./install.md#minikube).
  - [kops would launch a Kuberntes cluster on AWS](./install.md#aws).
  - We will also prepare a public Kubernetes cluster via Cloud if you don't have an AWS
    account that you can submit the EDL training jobs using the public cluster.

## Resources

- [PaddlePaddle](http://github.com/PaddlePaddle/Paddle)
- [PaddlePaddle EDL](https://github.com/PaddlePaddle/edl)

## Part-1: Train a Simple Model Using PaddlePaddle

In this part, we will train a **word embedding** model, to learn the concept of this
model, you can check out [word2vec](https://github.com/PaddlePaddle/book/tree/develop/04.word2vec), this 
tutorial will focus on writing the program.

### Training Codes

- Importing some necessary PaddlePaddle packages at the begging:

```python
import math
import os

import numpy
import paddle.v2 as paddle
```

- functions used to save and load word dict and embedding table

``` python
# save and load word dict and embedding table
def save_dict_and_embedding(word_dict, embeddings):
    with open("word_dict", "w") as f:
        for key in word_dict:
            f.write(key + " " + str(word_dict[key]) + "\n")
    with open("embedding_table", "w") as f:
        numpy.savetxt(f, embeddings, delimiter=',', newline='\n')


def load_dict_and_embedding():
    word_dict = dict()
    with open("word_dict", "r") as f:
        for line in f:
            key, value = line.strip().split(" ")
            word_dict[key] = int(value)

    embeddings = numpy.loadtxt("embedding_table", delimiter=",")
    return word_dict, embeddings

```

-  Map the $n-1$ words $w_{t-n+1},...w_{t-1}$ before $w_t$ to a D-dimensional vector though matrix of dimention $|V|\times D$ (D=32 in this example).

``` python
def wordemb(inlayer):
    wordemb = paddle.layer.table_projection(
        input=inlayer,
        size=embsize,
        param_attr=paddle.attr.Param(
            name="_proj",
            initial_std=0.001,
            learning_rate=1,
            l2_rate=0)
    return wordemb
```

- Define name and type for input to data layer.

``` python
paddle.init(use_gpu=False, trainer_count=1)
word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)
# Every layer takes integer value of range [0, dict_size)
firstword = paddle.layer.data(
    name="firstw", type=paddle.data_type.integer_value(dict_size))
secondword = paddle.layer.data(
    name="secondw", type=paddle.data_type.integer_value(dict_size))
thirdword = paddle.layer.data(
    name="thirdw", type=paddle.data_type.integer_value(dict_size))
fourthword = paddle.layer.data(
    name="fourthw", type=paddle.data_type.integer_value(dict_size))
nextword = paddle.layer.data(
    name="fifthw", type=paddle.data_type.integer_value(dict_size))

Efirst = wordemb(firstword)
Esecond = wordemb(secondword)
Ethird = wordemb(thirdword)
Efourth = wordemb(fourthword)
```

- Concatenate n-1 word embedding vectors into a single feature vector.

``` python
contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])
```

- Feature vector will go through a fully connected layer which outputs a hidden feature vector.

``` python
hidden1 = paddle.layer.fc(input=contextemb,
                          size=hiddensize,
                          act=paddle.activation.Sigmoid(),
                          layer_attr=paddle.attr.Extra(drop_rate=0.5),
                          bias_attr=paddle.attr.Param(learning_rate=2),
                          param_attr=paddle.attr.Param(
                                initial_std=1. / math.sqrt(embsize * 8),
                                learning_rate=1))
```

- Hidden feature vector will go through another fully connected layer, turn into a $|V|$ dimensional vector. At the same time softmax will be applied to get the probability of each word being generated.

``` python
predictword = paddle.layer.fc(input=hidden1,
                              size=dict_size,
                              bias_attr=paddle.attr.Param(learning_rate=2),
                              act=paddle.activation.Softmax())
```

- We will use cross-entropy cost function.

``` python
cost = paddle.layer.classification_cost(input=predictword, label=nextword)
```

- Create parameters, optimizer and trainer.

``` python
parameters = paddle.parameters.create(cost)
adagrad = paddle.optimizer.AdaGrad(
    learning_rate=3e-3,
    regularization=paddle.optimizer.L2Regularization(8e-4))
trainer = paddle.trainer.SGD(cost, parameters, adagrad)
```

Next, we will begin the training process. `paddle.dataset.imikolov.train(word_dict, N)`
and `paddle.dataset.imikolov.test(word_dict, N)` is our training and testing dataset.
Both of the function will return a reader: In PaddlePaddle, `reader` is a python function which
returns a Python iterator which output a single data instance at a time.

`paddle.batch` takes reader as input, outputs a **batched reader**: In PaddlePaddle, a reader
outputs a single data instance at a time but batched reader outputs a minibatch of data instances.

``` python
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)

    if isinstance(event, paddle.event.EndPass):
        result = trainer.test(
                    paddle.batch(
                        paddle.dataset.imikolov.test(word_dict, N), 32))
        print "Pass %d, Testing metrics %s" % (event.pass_id, result.metrics)
        with open("model_%d.tar"%event.pass_id, 'w') as f:
            trainer.save_parameter_to_tar(f)

trainer.train(
    paddle.batch(paddle.dataset.imikolov.train(word_dict, N), 32),
    num_passes=100,
    event_handler=event_handler)
```

- Run the Python program using Docker image `paddlepaddle/paddle:0.11.0`

Start training with the following command:

``` bash
cd example
docker run --rm -it -v $PWD:/work paddlepaddle/paddle:0.11.0 python work/train_local.py
```

The output of `event_handler` will be similar to following:

``` text
Pass 0, Batch 0, Cost 7.870579, {'classification_error_evaluator': 1.0}
Pass 0, Batch 100, Cost 6.052320, {'classification_error_evaluator': 0.84375}
Pass 0, Batch 200, Cost 5.795257, {'classification_error_evaluator': 0.8125}
Pass 0, Batch 300, Cost 5.458374, {'classification_error_evaluator': 0.90625}
```

After 30 passes, we can get an average error rate around 0.735611.

## Part-2: Launch the PaddlePaddle EDL Training Jobs on a Kubernetes Cluster

Before launching the EDL training-jobs, we can start-up a monitor program to
watch the Trainer process changes.

### Configure kubectl

If you start up a Kubernetes by `minikube` or `kops`, the kubectl configuration would be ready when
the cluster is available, for the other approach, you can contact the administrator to fetch the configuration file.

### Deploy EDL Components

**NOTE**: there is only one EDL controller in a Kubernetes cluster, so if you're using a public cluster, you can skip this step.

1. (Optional) Configure RBAC for EDL controller so that it would have the cluster admin permission.

If you launch a Kubernetes cluster by kops on AWS, the default authenticating policy is `RBAC`, so this step is **necessary**:

```bash
kubectl create -f k8s/rbac_admin.yaml
```

1. Create TRP "Training-Job"

As simple as running the following command:

``` bash
kubectl create -f k8s/thirdpartyresource.yaml
```

To verify the creation of the resource, run the following command:

``` bash
kubectl describe ThirdPartyResource training-job
```

- Deploy EDL controller

```bash
kubectl create -f k8s/edl_controller.yaml
```

### Launch the EDL Training Jobs

1. Edit the local training program to be able to run with distributed mode

It's easy to update your local training program to be running with distributing mode:

- Pre-process the datase with RecordIO format

We have done this in the Docker image `paddlepaddle/edl-example` using `dataset.covert` API as follows:

``` python
dataset.common.convert('/data/recordio/imikolov/', dataset.imikolov.train(word_dict, 5), 5000, 'imikolov-train')"
```

This would generate many recordio files on `/data/recordio/imikolov` folder, and we have prepared these files on Docker image `paddlepaddle/edl-example`.

- Pass in the `etcd_endpoint` to the `Trainer` object so that `Trainer` would know it's a fault-tolerant distributed training job.

``` python
trainer = paddle.trainer.SGD(cost,
                              parameters,
                              adam_optimizer,
                              is_local=False,
                              pserver_spec=etcd_endpoint,
                              use_etcd=True)
```

- Use `cloud_reader` which is a `master_client` instance can fetch the training data from the task queue.

``` python
trainer.train(
    paddle.batch(cloud_reader([TRAIN_FILES_PATH], etcd_endpoint), 32),
    num_passes=30,
    event_handler=event_handler)
```

1. Run the monitor program

Please open a new tab in your terminal program and run the monitor Python script `example/collector.py`:

```bash
docker run --rm -it -v $HOME/.kube/config:/root/.kube/config $PWD:/work paddlepaddle/edl-example python collector.py
```

And you can see the following metrics:

``` text
SUBMITTED-JOBS    PENDING-JOBS    RUNNING-TRAINERS    CPU-UTILS
0    0    -    18.40%
0    0    -    18.40%
0    0    -    18.40%
...
```

1. Deploy EDL Training Jobs

As simple as the following commands to launch a training-job on Kubernetes:

```bash
kubectl create -f example/examplejob.yaml
```

1. Deploy Multiple Training Jobs and Check the Monitor Logs

You can edit the YAML file and change the `name` field so that you can submit multiple training jobs.
For example, I submited three jobs which name is `example`, `example1` and `example2`, the monitor logs
is as follows:

``` text
SUBMITED-JOBS    PENDING-JOBS    RUNNING-TRAINERS    CPU-UTILS
0    0    -    18.40%
0    0    -    18.40%
1    1    example:0    23.40%
1    0    example:10    54.40%
1    0    example:10    54.40%
2    0    example:10|example1:5    80.40%
2    0    example:10|example1:8    86.40%
2    0    example:10|example1:8    86.40%
2    0    example:10|example1:8    86.40%
2    0    example:10|example1:8    86.40%
3    1    example2:0|example:10|example1:8    86.40%
3    1    example2:0|example:10|example1:8    86.40%
3    1    example2:0|example:5|example1:4    68.40%
3    1    example2:0|example:3|example1:4    68.40%
3    0    example2:4|example:3|example1:4    88.40%
3    0    example2:4|example:3|example1:4    88.40%
```

- At the begging, then there is no training job in the cluster except some Kubernetes system components, so the CPU utilization is **18.40%**.
- After submitting the training job **example**, the CPU utilization rise to **54.40%**, because of the `max-instances` in the YAML file is **10**, so the running trainers is **10**.
- After submitting the training job **example1**, the CPU utilization rose to **86.40%**.
- While we submitting the training job **example2**, there is no more resource for it, so EDL auto-scaller would
scale down the other jobs' trainer process, and eventually the running trainers of **example** dropped down to **3**, **example1** dropped down to **4** and no pending jobs.
