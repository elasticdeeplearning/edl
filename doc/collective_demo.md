# Introduction
This demo is for developers of EDL: you can test Paddle EDL function without a Kubernetes cluster. And it's simple to test it on a none or multiple nodes.
Of course, this is also a toy. You can play with it!
Have fun!

# Install
1. Install EDL from source

```
git clone https://github.com/PaddlePaddle/edl
cd edl
mkdir build & cd build
cmake ..
pip install ./python/dist/paddle_edl-0.0.0-cp27-cp27mu-linux_x86_64.whl
```

2. Install EDL using `pip install paddle_edl`.  

# Run the demo on a single node
1. Start a Jobserver on one node. 

```
git clone https://github.com/PaddlePaddle/edl
cd python/edl/demo/collective
./start_job_server.sh
```

2. Start a Jobclient on every node. Jobclient controls the POD process.

```
mkdir -p resnet50_pod
./start_job_client.sh
```

There is a bash script file `package.sh`. It's for preparing the environment for training.

  -  Set the ImageNet data path:  
  `export PADDLE_EDL_IMAGENET_PATH=<your path>`
  
  -  Set the `checkpoint` path:
  `export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>`