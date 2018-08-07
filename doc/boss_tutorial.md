# PaddlePaddle Tutorial for BOSS Workshop 2018

<img src="../logo/paddle.png" width="300">

PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use, efficient,
flexible and scalable deep learning platform, which is originally developed by
Baidu scientists and engineers for the purpose of applying deep learning to many
products at Baidu.

Fluid is the latest version of PaddlePaddle, it describes the model for training 
or inference using the representation of "Program".

PaddlePaddle Elastic Deep Learning (EDL) is a clustering project which leverages
PaddlePaddle training jobs to be scalable and fault-tolerant.
EDL will greatly boost the parallel distributed training jobs and make good use
of cluster computing power.

EDL is based on the full fault-tolerant feature of PaddlePaddle, it uses a Kubernetes controller
to manage the cluster training jobs and an auto-scaler to scale the job's computing resources.

## Tutorial Outline

- Introduction

    At the introduction session, we will introduce:
    - PaddlePaddle Fluid design overview.
    - Fluid Distributed Training.
    - Why we develop PaddlePaddle EDL and how we implement it.

- Hands-on Tutorial

    We have some hands-on tutorials after each introduction
    session so that all the audience can use PaddlePaddle and ask some questions
    while using PaddlePaddle:
    - Training models using PaddlePaddle Fluid in a Jupyter Notebook (PaddlePaddle Book).
    - Laucnh a Distributed Training Job on your laptop.
    - Launch the EDL training job on a Kubernetes cluster.

- Intended audience

    People who are interested in deep learning system architecture.
  

## Prerequisites

- [Install Docker](https://docs.docker.com/install/)
- [Install kubectl](./install.md#kubectl)
- [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- A Kubernetes cluster which version is `1.7.x`:
    - [minikube would launch a kubernetes cluster locally](./install.md#minikube).
    - [kops would launch a Kuberntes cluster on AWS](./install.md#aws).
    - We will also prepare a public Kubernetes cluster via Cloud if you don't have an AWS
      account that you can submit the EDL training jobs using the public cluster.

## Resources

- [PaddlePaddle](http://github.com/PaddlePaddle/Paddle)
- [PaddlePaddle Book](http://github.com/PaddlePaddle/book)
- [PaddlePaddle EDL](https://github.com/PaddlePaddle/edl)

## Part-1 Training Models on Your Laptop using PaddlePaddle

### PaddlePaddle Book

Please checkout [PaddlePaddle Book](http://github.com/PaddlePaddle/book), steps to run
the training process and example output.

### Launch a Distributed Training Job on Your Laptop

1. Launch the PaddlePaddle Production Docker Container:

    ``` bash
    > git clone https://github.com/PaddlePaddle/edl.git
    > cd edl/example/fluid
    > docker run --name paddle -d -it -v $PWD:/work paddlepaddle/paddle /bin/bash
    ```

1. Split training data into multiple parts:

    ``` python
    > docker exec -it paddle /bin/bash
    > cd work
    > python dist_word2vec.py prepare
    ```

    would split the `imikolov` data into multiple parts like:

    ``` bash
    ./output/
    ./output/mnist-train-00000.pickle
    ./output/mnist-train-00001.pickle
    ./output/mnist-train-00002.pickle
    ./output/mnist-train-00003.pickle
    ./output/mnist-train-00004.pickle
    ...
    ```

1. Luanch **two** PServer instances and **two** Trainer instances:

  Start PServer instance:

  ``` python
  > PADDLE_PSERVER_EPS=127.0.0.1:6789 PADDLE_TRAINERS=2 \
    PADDLE_TRAINING_ROLE=pserver PADDLE_CURRENT_ENDPOINT=127.0.0.1:6789 \
    python dist_word2vec.py train
  ```

  Start Trainer instance which `trainer_id=0`:

  ``` python
  > PADDLE_PSERVER_EPS=127.0.0.1:6789 PADDLE_TRAINERS=2 \
    PADDLE_TRAINING_ROLE=trainer PADDLE_TRAINER_ID=0 python dist_word2vec.py train
  ```

  Start Trainer instance which `trainer_id=1`:

  ``` python
  > PADDLE_PSERVER_EPS=127.0.0.1:6789 PADDLE_TRAINERS=2 \
    PADDLE_TRAINING_ROLE=trainer PADDLE_TRAINER_ID=1 python dist_word2vec.py train
  ```

## Part-2: Launch the PaddlePaddle EDL Training Jobs on a Kubernetes Cluster

Please note, EDL only support the early PaddlePaddle version so the fault-tolerant model is
written with PaddlePaddle v2 API.

### Configure kubectl

If you start up a Kubernetes instance by `minikube` or `kops`, the kubectl configuration would be ready when
the cluster is available, for the other approach, you can contact the administrator to fetch the configuration file.

### Deploy EDL Components

**NOTE**: there is only one EDL controller in a Kubernetes cluster, so if you're using a public cluster, you can skip this step.

1. (Optional) Configure RBAC for EDL controller so that it would have the cluster admin permission.

    If you launch a Kubernetes cluster by kops on AWS, the default authenticating policy is `RBAC`, so this step is **necessary**:

    ```bash
    kubectl create -f k8s/rbac_admin.yaml
    ```

1. Create TPR "Training-Job"

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

    - Dataset

        Pre-process the dataset to convert to RecordIO format,
        We have done this in the Docker image `paddlepaddle/edl-example` using `dataset.covert` API as follows:

        ``` python
        dataset.common.convert('/data/recordio/imikolov/', dataset.imikolov.train(word_dict, 5), 5000, 'imikolov-train')"
        ```

        This would generate many recordio files on `/data/recordio/imikolov` folder, and we have prepared these files on Docker image `paddlepaddle/edl-example`.

    - Pass the `etcd_endpoint` to the `Trainer` object so that `Trainer` would know it's a fault-tolerant distributed training job.

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
