# EDL: Elastic Deep Learning

<img src="https://github.com/elasticdeeplearning/artwork/blob/master/horizontal/color/edl-horizontal-color.png" width="500" style="display:inline;vertical-align:middle;padding:2%">

# Motivation
Elastic Deep Learning(EDL) is a framework with the ability to dynamically adjust the parallelism (number of training workers) for deep neural network training. It can support multi-tenant cluster management to balance job completion time and job waiting time, maximize the use of idle resources, and so on.

This project contains EDL framework and its applications such as distillation and NAS.

Now EDL is an incubation-stage project of the [LF AI Foundation](https://lfai.foundation).

<img src="https://github.com/lfai/artwork/blob/master/lfai-project-badge/incubation/color/lfai-projectlogos_incubation-color.png"  width="200" style="display:inline;vertical-align:middle;padding:2%">

# Installation
You can install with ```pip install paddle_edl```. But we highly **recommend** you use it in our docker:

```
docker pull hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7
nvidia-docker run -name paddle_edl hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7 /bin/bash
```  

# EDL Applications:
## EDL Distill Training
<p align="center">
    <img src="doc/distill.gif" width="700">
</p>
The distillation training consists of two parts, teacher and student. Generally, the teacher and student are placed on the same card.
EDL distillation training uncouple teacher and student, treats teacher as a inference service.
Students send sample data to the teacher and obtain prediction results for training.
Prediction result acquisition and packaging is currently done by the DistillReader of the EDL.

### How to change from a normal train program to an EDL distill train program
1. Define an input to represent the variable obtained from the teacher.
2. Use DistillReader to define the original reader input and the variables that need to be obtained from the teacher.
Then set train_reader as the data source of DistillReader.
3. Use student's prediction and teacher's prediction to define loss function.
``` python
# 1. define an input represent teacher prediction
soft_label = fluid.data(name='soft_label', shape=[None, 10], dtype='float32')
inputs.append(soft_label)

# 2. define DistillReader
dr = DistillReader(ins=['img', 'label'], predicts=['fc_0.tmp_2'])
train_reader = dr.set_sample_list_generator(train_reader)

# 3. define distill loss
distill_loss = fluid.layers.cross_entropy(
    input=prediction, label=soft_label, soft_label=True)
distill_loss = fluid.layers.mean(distill_loss)
loss = distill_loss

# Start distill train.
# data includes the original reader input and the prediction results obtained from the teacher,
# that is (img, label, soft_label)
for data in train_reader():
    metrics = exe.run(main_program, feed=data, fetch_list=[loss, acc])
```

### Quick Start
#### Run with fixed teacher
**A complete example is [here](./mnist_distill).**

EDL distill relay on paddle-serving, please install with 
```pip install paddle-serving-client paddle-serving-server-gpu```
1. First, you need deploy teacher. We using [Paddle Serving](https://github.com/PaddlePaddle/Serving) to deploy teacher.
You can see [here](https://github.com/PaddlePaddle/Serving/blob/develop/doc/SAVE.md) to save your own teacher model.
``` bash
cd example/distill/mnist_distill
wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/mnist_cnn_model.tar.gz
python -m paddle_serving_server_gpu.serve \
  --model mnist_cnn_model \
  --port 9292 \
  --gpu_ids 0
```
2. Prepare student code. Use `set_fixed_teacher` to set fixed teacher.
``` python
# see example/distill/mnist_distill/train_with_fleet.py
dr = DistillReader(ins=['img', 'label'], predicts=['fc_0.tmp_2'])
dr.set_fixed_teacher(args.distill_teachers)
train_reader = dr.set_sample_list_generator(train_reader)
```
The run student code.
``` python
python train_with_fleet.py \
  --use_distill_service True \
  --distill_teachers 127.0.0.1:9292
```

Run with dynamic teacher need deploy discovery service, please see 
[Run EDL distillation training demo on Kubernetes or a single node](./example/distill/README.md)

# EDL Framework
## How to change from a normal train program to an EDL train program
The main change is that you should `load_checkpoint` at the beginning of training and `save_checkpoint` at the end of every epoch and the checkpoint should be on a distributed file system such as HDFS so all trainers can download from it. A complete example is [here](https://github.com/elasticdeeplearning/edl/tree/develop/example/collective/resnet50)

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

## Quickstart
### EDL Resnet50 experiments on a single machine in docker:

1. Start a JobServer on one node which generates changing scripts.
 
```
cd example/demo/collective	
./start_job_server.sh
```

1. Start a Jobclient which controls the worker process.

```
#Set the ImageNet data path
export PADDLE_EDL_IMAGENET_PATH=<your path>
#Set the checkpoint path
export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>

mkdir -p resnet50_pod
./start_job_client.sh
```

1. Experiments result
 
| total batch size | acc1 | acc5 |
| :-----: | ----: | ----: |
| 1024 | 75.5 | 92.8 |


## FAQ

TBD

## License

EDL is provided under the [Apache-2.0 license](LICENSE).
