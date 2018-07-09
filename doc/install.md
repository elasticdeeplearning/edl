# Install EDL dependencies

Before everything, make sure you have a running Kubernetes v1.7.* cluster and a working `kubectl`.
You may prepare a Kubernetes on your laptop using [minikube](#minikube), create a Kubernetes Cluster on [AWS](#AWS)
or any other way as you expediently.

## kubectl

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

- Windows

```bash
wget -O kubectl.exe https://storage.googleapis.com/kubernetes-release/release/v1.9.0/bin/windows/amd64/kubectl.exe
```

And then add the binary file to your PATH.

## Kubernetes Cluster

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
an easy way to get a production grad Kubernetes on AWS.

You can checkout the [tutorial](https://github.com/kubernetes/kops/blob/master/docs/aws.md) for the details of launching a Kubernetes cluster on AWS.

- Prepare kops environment

- Configurate DNS

Because of I bought a domain `yancey.co.uk` via AWS, and had a hosted zone in Route53, so I don't need to do more work about DNS configuration,
for the other condition, please following [Configure DNS](https://github.com/kubernetes/kops/blob/master/docs/aws.md#configure-dns).

Please testing your DNS setup before continue:

```bash
dig ns yancey.co.uk
```

Should return something similar to:

```bash
yancey.co.uk.		172778	IN	NS	ns-1939.awsdns-50.co.uk.
yancey.co.uk.		172778	IN	NS	ns-1101.awsdns-09.org.
yancey.co.uk.		172778	IN	NS	ns-320.awsdns-40.com.
yancey.co.uk.		172778	IN	NS	ns-563.awsdns-06.net.
```

**NOTE**: DNS is a critical component for `kops`, please validated your DNS configuration before moving on.

- Cluster State Storage

```bash
aws s3api create-bucket \
    --bucket prefix-example-com-state-store \
    --region us-east-1
```

- Prepare local environment

```bash
export NAME=myfirstcluster.example.com
export KOPS_STATE_STORE=s3://prefix-example-com-state-store

```

- Create your cluster

We will need to note which availability zones are available to us. In this example we will be deploying our cluster to the us-west-2 region.

```bash
aws ec2 describe-availability-zones --region us-west-2
```

Then you can use the following command to create the cluster:

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

It would take serival minutes to create the cluster ...

- Validating the cluster

You can use the following command to validate the cluster:

```bash
kops validate cluster
```

When you see the output like:

```text

Validating cluster kops.example.yancey.co.uk

INSTANCE GROUPS
NAME			ROLE	MACHINETYPE	MIN	MAX	SUBNETS
master-us-west-2a	Master	t2.medium	1	1	us-west-2a
nodes			Node	t2.xlarge	3	3	us-west-2a

NODE STATUS
NAME						ROLE	READY
ip-172-20-35-244.us-west-2.compute.internal	node	True
ip-172-20-54-167.us-west-2.compute.internal	node	True
ip-172-20-61-99.us-west-2.compute.internal	master	True
ip-172-20-62-223.us-west-2.compute.internal	node	True

Your cluster kops.example.yancey.co.uk is ready
```

That means your cluster is ready

- Configure RBAC for your cluster

The default authauthentication policy of the Kubernetes cluster created by `kops` is `RBAC`,
to allow EDL controller changing the number of trainer process, we need to assign the controller `admin`
permission.

```bash
kubectl create ../k8s/admin.yaml
```

- Delete you cluster

You can execute the following command the delete the cluster configuration and resource that
you created above.

```bash
kops delete cluster ${NAME} --yes
```
