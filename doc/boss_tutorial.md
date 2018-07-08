# PaddlePaddle EDL tutorial

This is a hands-on tutorial used for BOSS workshop 2018. We have two parts in this tutorial:
- Part-1: Train a simple model by PaddlePaddle Fliud.
- Part-2: Lunch the EDL training job on a Kuberntes cluster.

And please don't forget the necessary preparations before everything.

## Prerequisites

- Install Docker
- Install kubectl
- A production-ready Kubernetes cluster which version is `1.7.x`
  - minikube would lunch a kubernetes cluster locally.
  - kops would lunch a Kuberntes cluster on AWS.

Please note, TPR (Third Party Resource) is deprecated after Kubernetes 1.7. We are working to support CRD (Custom Resource Definitions, the successor of TPR). Stay tuned!

## Part-1: Train a simple model by PaddlePaddle Fliud

In this part, we will train a model from a realistic dataset to predict home prices,
you can check out [fit a line](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line) to get
some concept about Deep Learning, this tutorial would only force on codes:

### Write the codes to train the model

- Importing some necessary PaddlePaddle packages at the begging:

```python
import paddle
import paddle.fluid as fluid
import numpy
```

- Define data feeders for test and train

The feeder reads a BATCH_SIZE of data each time and feed them to the training/testing process.
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

### Run the Python program

```bash
docker run --rm -it -v $PWD:/work paddlepaddle/paddle:0.14.0 python /work/example/train_fluid.py
```

The training process could take up to a few minutes, you can see the training logs
in the meantime which defined in `EventHandler`:

```text
```

## Part-2: Lunch the PaddlePaddle EDL Training Jobs on a Kubernetes Cluster

Before launching the EDL training-jobs, we can start-up a monitor program to
watch the Trainer process changes.

1. Configure kubectl

If you start up a Kubernetes by `minikube` or `kops`, the kubectl configuration would be ready when
the cluster is ready, for the other approach, you can contact the administrator to fetch the configuration file.

1. (Optional) Configure RBAC for EDL controller so that it would have the cluster admin permission.

If you lunch a Kubernetes cluster by kops on AWS, the default authenticating policy is `RBAC`,
so this step is **necessary**:

```bash
kubectl create -f kops/paddlecloud_admin.yaml
```

1. Create TRP "TrainingJobs"

As simple as running the following command

``` bash
kubectl create -f ../k8s/thirdpartyresource.yaml
```

To verify the creation of the resource, run the following command:

``` bash
kubectl describe ThirdPartyResource training-job
```

1. Deploy EDL controller

1. Deploying an EDL training job which name is `example`

You can see the logs in the monitor program



```text
```

1. Deploying another EDL training-job which name is `example2`


```text

```