<p align="left">
    <br>
<img src='https://github.com/elasticdeeplearning/artwork/blob/master/horizontal/color/edl-horizontal-color.png' width = "450">
    <img src='https://github.com/lfai/artwork/blob/master/lfai-project-badge/incubation/color/lfai-projectlogos_incubation-color.png' width = "200">
    <br>
<p>

<h2 align="center">Motivation</h2>

Computing resources on cloud such as [Amazon AWS](https://aws.amazon.com/cn/)、[Baidu Cloud](https://cloud.baidu.com/) have multi-tenancy. Deep learning model training and inference with elastic resources will be common on cloud. We propose Elastic Deep Learning (EDL) that makes training and inference of deep learning model on cloud easier and more efficient.

Now EDL is an incubation-stage project of the [LF AI Foundation](https://lfai.foundation).


<h2 align="center">Installation</h2>

You can install with ```pip install paddle_edl```. But we highly **recommend** you use it in our docker:

```
docker pull hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7
nvidia-docker run -name paddle_edl hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7 /bin/bash
```  

<h2 align="center">Latest Release(0.3.0)</h2>

- Support elastic training with inference type services during training, such as knowledge distillationo 
- Inference type services are automatically registered through service discovery in EDL
- Knowledge distillation examples in computer vision and natural language processing

<h3 align="center">Quick start on a signal machine</h3>

- The Teacher Model: [ResNeXt101_32x16d_wsl](https://github.com/facebookresearch/WSL-Images). Start ResNeXt101_32x16d_wsl teacher on gpu 1.
``` bash
cd example/distill/resnet

wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/ResNeXt101_32x16d_wsl_model.tar.gz
tar -zxf ResNeXt101_32x16d_wsl_model.tar.gz

python -m paddle_serving_server_gpu.serve \
  --model ResNeXt101_32x16d_wsl_model \
  --port 9898 \
  --gpu_ids 1
```

- The Student Model: [ResNet50_vd](https://arxiv.org/pdf/1812.01187.pdf)(that is ResNet-D in paper). Train ResNet50_vd student on gpu 0.
``` bash
python -m paddle.distributed.launch --selected_gpus 0 \
  ./train_with_fleet.py \
  --model=ResNet50_vd \
  --data_dir=./ImageNet \
  --use_distill_service=True \
  --distill_teachers=127.0.0.1:9898
```

- Performance comparison

| total batch size | acc1 | acc5 |
| :-----: | ----: | ----: |
| 512 | 79.1 | 94.4 |

<h3 align="center">About Knowledge Distillation in EDL</h3>

- Theory: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
    - Knowledge distillation consists of two parts in general, i.e. strong teachers and weak students. 
    - Student model learns from a teacher or mixture-of-teachers model's feed-forward results to achieve better results.
- Application scenarios of EDL knowledge distillation
    - Teacher models and student models are runing on the same GPU devices that training throughputs are not maximized
    - Offline GPU cluster has limited resources but some online GPU resources can be used during training.
    - Heterogenous teacher models can improve student model's performance but are hard to deploy on single GPU card due to memory limitation.
    - Computation burden of teacher models and student models are hard to balanced to maximize the training throughputs.
- Solution:     
    - Deploy teacher models as online inference service through [Paddle Serving](https://github.com/PaddlePaddle/Serving)
    - Online inference services are elastic and are registered to EDL service management modules.
    - Dynamical adaptation of teacher models' online instance to maximize students' training throughputs and resources utilization.

<p align="center">
    <img src="doc/distill.gif" width="550">
</p>

- To run distillation on clusters, please reference [Run EDL distillation training](./example/distill/README.md)

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
