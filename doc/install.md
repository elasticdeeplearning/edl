# Install EDL dependencies

## Kubernetes

Before everything, make sure you have a running Kubernetes v1.7.* cluster and a working `kubectl`.
You may prepare a Kubernetes on your laptop using [minikube](#minikube), create a Kubernetes Cluster on [AWS](#AWS)
or any other way as you expediently.

### kubectl

`kubectl` is the CLI tool to manager your Kubernetes cluster, and kubectl `v1.9.0` works well for Kubernets v1.7.*,
you can install it by following commands:

- MacOS

```bash
wget -O kubectl https://storage.googleapis.com/kubernetes-release/release/v1.9.0/bin/darwin/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```

- Linux

```bash
wget -O kubectl https://storage.googleapis.com/kubernetes-release/release/v1.9.0/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
```

### minikube

If you just trying to play EDL in your laptop, go with [minikube](https://github.com/kubernetes/minikube) with the following command is good enough to get you ready.

```bash
minikube start --kubernetes-version v1.7.16
```

To verify your `minikube` and `kubectl` works, run the following command:

``` bash
kubectl version
```

### AWS

If you are trying to deploy EDL on a real Kubernetes cluster, [kops](https://github.com/kubernetes/kops) is
an easy way to get a production grad Kubernetes.

You can checkout the [tutorial](https://github.com/kubernetes/kops/blob/master/docs/aws.md#configure-dns) for the details of lunching a Kubernetes cluster on AWS.

You can use the following command to create the cluster instead of the simple one in the tutorial:

```bash
kops create cluster \
    --node-count 3 \
    --zones us-west-2a \
    --master-zones us-west-2a \
    --node-size t2.medium \
    --master-size t2.medium \
    --topology private \
    --networking weave \
    --cloud-labels "Team=Paddle,Owner=Paddle" \
    --kubernetes-version=1.7.16 \
    ${NAME} --yes
```

- Configure RBAC for your cluster

```bash
kubectl create ../k8s/admin.yaml
```

- Delete you cluster

```bash
kops delete cluster ${NAME} --yes
```
