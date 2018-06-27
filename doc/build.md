# How to Build EDL Component

This article contains instructions of build EDL and how to pack them into
Docker image so that the EDL component can run in the Kubernetes cluster.

## Build EDL Controller

```bash
glide install --strip-vendor
go build github.com/paddlepaddle/edl/cmd/edl
```

The above step will generate a binary file named `edl` which should
run as a daemon process on the Kubernetes cluster.

## Build EDL Controller Image

To build your own docker images, run the following command:

```bash
docker build -t yourRepoName/edl-controller .
```

This command will take the `Dockerfile`, build the EDL docker image and tag it as `yourRepoName/edl-controller`

Now you want to push it to your docker hub so that Kubernetes cluster is able to pull and deploy it.

``` bash
docker push yourRepoName/edl-controller
```
