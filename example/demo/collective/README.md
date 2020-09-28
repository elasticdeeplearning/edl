# Purpose
This article illustrates how to change the train program to an EDL program, and run on single or multiple nodes.


## How to change from a normal train program to an EDL train program
The main changes are:

- `load_checkpoint`  should be added at the beginning of training and
- `save_checkpoint` added at the end of every epoch.  
   the checkpoint should be on a distributed file system such as HDFS so all trainers can download from it. A complete example is [here](https://github.com/elasticdeeplearning/edl/tree/develop/example/collective/resnet50)

```
fs=HDFSClient(args.hdfs_name, args.hdfs_ugi,20*60*1000, 3 * 1000)

train_status =TrainStatus()
tmp_s = fleet.load_checkpoint(exe, args.checkpoint, fs=fs, trainer_id=trainer_id)
if tmp_s is not None:
   train_status = tmp_s

for pass_id in range(train_status.next(), params["num_epochs"]):
    train()

    if trainer_id == 0:
        saved_status = TrainStatus(pass_id)  
        fleet.save_checkpoint(exe, train_status=saved_status,
            path=args.checkpoint, fs=fs)
```

The epoch's number is stored in `train_status` and the epoch number will be restored when the checkpoint is loaded.

## Start Resnet50 demo training multiple nodes:

1. Start a JobServer on one node which generates changing scripts.

```
node_ips="192.168.10.1,192.168.10.2"
python -u paddle_edl.demo.collective.job_server_demo \
    --node_ips ${node_ips} \
    --pod_num_of_node 8 \
    --time_interval_to_change 900 \
    --gpu_num_of_node 8
```

1. Start a Jobclient on every node which controls the worker process.

```
# set the ImageNet data path
export PADDLE_EDL_IMAGENET_PATH=<your path>
# set the checkpoint path
export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>
export PADDLE_JOBSERVER="http://192.168.10.1:8180"

mkdir -p resnet50_pod
unset http_proxy https_proxy

# running under edl
export PADDLE_RUNING_ENV=PADDLE_EDL
export PADDLE_JOB_ID="test_job_id_1234"
export PADDLE_POD_ID="not set"

python -u paddle_edl.demo.collective.job_client_demo \
    --log_level 20 \
    --package_sh ./resnet50/package.sh \
    --pod_path ./resnet50_pod \
    ./train_pretrain.sh
```


## On Kubernetes

We have built the docker images for you and you can start a demo on Kubernetes immediately:
TBD
