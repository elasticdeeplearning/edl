# PaddlePaddle Elastic Deep Learning

<img src="../logo/edl.png" width="500">

While many hardware and software manufacturers are working on improving the training performance
of deep learning jobs, PaddlePaddle Elastic Deep Learning (EDL) optimize the global utilization
of the cluster and the waiting time of job submitters. EDL includes two parts, one is a
Kubernetes controller, PaddlePaddle auto-scaler which changes the number of processes of the 
distributed jobs according to the idle hardware resource in the cluster,
and another one is the fault-tolerant distributed training architecture in PaddlePaddle.

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

## Part-1: Train a Simple Model Using Paddlepaddle Fliud

In this part, we will train a model from a real dataset to predict house prices,
to learn the concept of this model, you can check out
[fit a line](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line),
this tutorial will focus on writing the program.

### Write the Codes to Train a Simple Model

- Importing some necessary PaddlePaddle packages at the begging:

```python
import paddle
import paddle.fluid as fluid
import numpy
```

- Define data feeders for test and train

The feeder reads a BATCH_SIZE of data each time and feeds them to the training/testing process.
If the user wants some randomness on the data order, she can define both a BATCH_SIZE and a buf_size.
That way the data feeder will yield the first BATCH_SIZE data out of a shuffle of the first buf_size data.

```python
BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE)
```

- Train Program Configuration

`train_program` sets up the network structure of this current training model.
For linear regression, it is merely a fully connected layer from the input to the output.
The train_program must return an avg_loss as its first returned parameter because it is needed in backpropagation.

```python
def train_program():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    # feature vector of length 13
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)

    return avg_loss
```

- Optimizer Function Configuration

In the following SGD optimizer, learning_rate specifies the learning rate in the optimization procedure.

```python
def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.001)
```

- Specify Place

Specify your training envionment if the training is on CPU or GPU: 

```bash
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
```

- Crete Trainer Object

The trainer will take the train_program as input.

```python
trainer = fluid.Trainer(
    train_func=train_program,
    place=place,
    optimizer_func=optimizer_program)
```

- Feeding Data

PaddlePaddle provides the reader mechanism for loading the training data.
A reader may return multiple columns, and we need a Python dictionary to specify
the mapping from column index to data layers.

```python
feed_order=['x', 'y']
```

- Event Handler

An event handler is provided to print the training progress:

```python
# Specify the directory path to save the parameters
params_dirname = "fit_a_line.inference.model"

# event_handler to print training and testing info
def event_handler_plot(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 10 == 0: # every 10 batches, record a test cost
            test_metrics = trainer.test(
                reader=test_reader, feed_order=feed_order)

            if test_metrics[0] < 10.0:
                # If the accuracy is good enough, we can stop the training.
                print('loss is less than 10.0, stop')
                trainer.stop()

        # We can save the trained parameters for the inferences later
        if params_dirname is not None:
            trainer.save_params(params_dirname)

        step += 1
```

- Start Training

The following codes would start the training as the belove configuration:

```python
trainer.train(
    reader=train_reader,
    num_epochs=100,
    event_handler=event_handler_plot,
    feed_order=feed_order)
```

### Run the Python Program Using Docker

```bash
docker run --rm -it -v $PWD:/work paddlepaddle/paddle:0.14.0 python /work/example/train_fluid.py
```

The training process could take up to a few minutes, and you can see the training logs
in the meantime which defined in `EventHandler`:

```text
```

## Part-2: Launch the Paddlepaddle EDL Training Jobs on a Kubernetes Cluster

Before launching the EDL training-jobs, we can start-up a monitor program to
watch the Trainer process changes.

### Configure kubectl

If you start up a Kubernetes by `minikube` or `kops`, the kubectl configuration would be ready when
the cluster is available, for the other approach, you can contact the administrator to fetch the configuration file.

### Deploy EDL Components

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
