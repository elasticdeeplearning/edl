# PaddlePaddle EDL: Elastic Deep Learning

<img src="logo/edl.png" width="500">

While many hardware and software manufacturers are working on
improving the running time of deep learning jobs, EDL optimizes

1. the global utilization of the cluster, and
1. the waiting time of job submitters.

For more about the project EDL, please refer to this [invited blog
post](http://blog.kubernetes.io/2017/12/paddle-paddle-fluid-elastic-learning.html)
on the Kubernetes official blog.

EDL includes two parts:

1. a Kubernetes controller for the elastic scheduling of distributed
   deep learning jobs, and

1. making PaddlePaddle a fault-tolerable deep learning framework.
   This directory contains the Kubernetes controller.  For more
   information about fault-tolerance, please refer to the
   [design](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/cluster_train).

We deployed EDL on a real Kubernetes cluster, dlnel.com, opened for
graduate students of Tsinghua University.  The performance test report
of EDL on this cluster is
[here](https://github.com/PaddlePaddle/cloud/blob/develop/doc/edl/experiment/README.md).


## Build

```bash
glide install --strip-vendor
go build -o path/to/output github.com/paddlepaddle/edl/cmd/edl
```

## Usage

To deploy the EDL to your kubernetes cluster, there are 2 major steps:

1. Create a Third Party Resource "Training-job" to allow creating a PaddlePaddle machine learning job in one yaml file.
1. Deploy the EDL controller to monitor and control overall cluster resource distribution between the online services and the PaddlePaddle training-jobs.

Please note, TPR (Third Party Resource) is deprecated after Kubernetes 1.7. We are working to support CRD (Custom Resource Definitions, the successor of TPR). Stay tuned!

### Prepare your cluster
So before everything, make sure you have a running Kubernetes v1.7.* cluster and a working `kubectl`.

If you just trying to play EDL in your laptop, go with `minikube` with the following command is good enough to get you ready.

``` bash
minikube start --kubernetes-version v1.7.5
```

To verify your `minikube` and `kubectl` works, run the following command:

``` bash
kubectl version
```

if you are able to see both client and server version, AND server version is v1.7.5, you are good to go.

### Create TPR "Training-job"

As simple as running the following command

``` bash
kubectl create -f thirdpartyresource.yaml
```

To verify the creation of the resource, run the following command:

``` bash
kubectl describe ThirdPartyResource training-job
```

if there is no error returned, that means your training-job TPR is successfully created.

### Deploy EDL controller

EDL is supposed to run as a docker images to run in the Kubernetes cluster in most of the cases, but it's always possible to run the EDL binary outside the cluster along with Kubernetes config file. In this section we will assume that the EDL runs as docker image in the Kubernetes cluster.

Before we get to the docker image part, we recommend running the EDL controller within a Kubernetes namespace, which provides better isolation among apps. By default, the EDL runs under namespace "paddlecloud". To create it, run the following command if you don't have it created.

``` bash
kubectl create namespace paddlecloud
```

There are 2 ways to obtain the EDL docker image:

1. Directly pull the pre-built image from docker hub's paddle repo
1. Build your own

If you decide to use the pre-built image, there is nothing you need to do now, you can skip to the deployment part.

To build your own docker images, run the following command:

``` bash
docker build -t yourRepoName/edl-controller .
```

This command will take the `Dockerfile`, build the EDL docker image and tag it as `yourRepoName/edl-controller`

Now you want to push it to your docker hub so that Kubernetes cluster is able to pull and deploy it.

``` bash
docker push yourRepoName/edl-controller
```
Before deploying your EDL controller, open `edl_controller.yaml` with any text editor to change the docker image uri from `paddlepaddle/edl-controller` to `yourRepoName/edl-controller`

Now let's deploy the EDL controller:

``` bash
kubectl create -f edl_controller.yaml
```

To verify the deployment, let's firstly verify the depolyment's pod is successfully created:

``` bash
kubectl get pods --namespace paddlecloud

NAME                                       READY     STATUS    RESTARTS   AGE
training-job-controller-2033113564-w80q6   1/1       Running   0          4m
```
Wait until you see `STATUS` is `Running`, run the following command to see controller's working log:

``` bash
kubectl logs training-job-controller-2033113564-w80q6 --namespace paddlecloud
```

when you see logs like this:

``` text
t=2018-03-13T22:13:19+0000 lvl=dbug msg="Cluster.InquiryResource done" resource="{NodeCount:1 GPURequest:0 GPULimit:0 GPUTotal:0 CPURequestMilli:265 CPULimitMilli:0 CPUTotalMilli:2000 MemoryRequestMega:168 MemoryLimitMega:179 MemoryTotalMega:1993 Nodes:{NodesCPUIdleMilli:map[minikube:1735] NodesMemoryFreeMega:map[minikube:1824]}}" stack="[github.com/paddlepaddle/edl/pkg/autoscaler.go:466 github.com/paddlepaddle/edl/pkg/controller.go:72]"
```
That means your EDL controller is actively working monitoring and adjusting resource distributions.

## Deploying a training-job

Now we have a resource typed `training-job` defined in Kubernetes and we have the EDL watching and optimizing the resource distribution, let's create a training job to see how it works.

Firstly, let's create your training job's docker image, which contains training logic in `example/train_ft.py`

``` bash
cd example
docker build -t yourRepoName/my_edl_training_job .
```

then push it to docker hub to be accessible by Kubernetes:

``` bash
docker push yourRepoName/my_edl_training_job
```

Please note, `docker build` uses `Dockerfile` in `example` directory, which indicates our `my_edl_training_job` is based on docker image `paddlepaddle/paddlecloud-job`. This images has PaddlePaddle installed and configured, so that you do not have to install on your own.

Now we have defined "What to run" for Kubernetes, it's time to define "How to run" the training job, which is supposed to configured in a yaml file. please find the example yaml definition of a training job from `example/examplejob.yaml`.

In this file, change the image uri from `paddlepaddle/paddlecloud-job` to `yourRepoName/my_edl_training_job` in this case.

In `spec` section you will see 2 major members `trainer` and `pserver`, their configurations are trying to define how "distributed" this job is. Like trainer and pserver 's `min-instance` and `max-instance` are showing the desired trainer count range, so that EDL will adjust the instance count based on these information. We'll have a separate document to describe these fields soon.

Now let's start the training job by run command below:

``` bash
kubectl create -f example.yaml
```

## Resource Adjustments by EDL

TBD

## FAQ

TBD

## License

PaddlePaddle EDL is provided under the [Apache-2.0 license](LICENSE).