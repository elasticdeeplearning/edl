<p align="left">
    <br>
<img src='https://github.com/elasticdeeplearning/artwork/blob/master/horizontal/color/edl-horizontal-color.png' width = "450">
    <img src='https://github.com/lfai/artwork/blob/master/lfai-project-badge/incubation/color/lfai-projectlogos_incubation-color.png' width = "200">
    <br>
<p>

<h2 align="center">Motivation</h2>

Computing resources on cloud such as [Amazon AWS](https://aws.amazon.com/cn/)、[Baidu Cloud](https://cloud.baidu.com/) have multi-tenancy. Deep learning model training and inference with elastic resources will be common on cloud. We propose Elastic Deep Learning (EDL) that makes training and inference of deep learning models on cloud easier and more efficient.

Now EDL is an incubation-stage project of the [LF AI Foundation](https://lfai.foundation).


<h2 align="center">Installation</h2>

You can install with ```pip install paddle_edl```. But we highly **recommend** you use it in our docker:

```
docker pull hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7
nvidia-docker run -name paddle_edl hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda9.0-cudnn7 /bin/bash
```  

<h2 align="center">Latest Release(0.3.0)</h2>

- Support elastic training with inference type services during training, such as knowledge distillation 
- Inference type services are automatically registered through service discovery in EDL
- Knowledge distillation examples in computer vision and natural language processing

<h3 align="center">Quick start Demo</h3>

- Install Paddle Serving

``` bash
pip install paddle-serving-server-gpu
```

- The Teacher Model: [ResNeXt101_32x16d_wsl](https://github.com/facebookresearch/WSL-Images). Start teacher on gpu 1.

``` bash
cd example/distill/resnet

wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/ResNeXt101_32x16d_wsl_model.tar.gz
tar -zxf ResNeXt101_32x16d_wsl_model.tar.gz

python -m paddle_serving_server_gpu.serve \
  --model ResNeXt101_32x16d_wsl_model \
  --mem_optim True \
  --port 9898 \
  --gpu_ids 1
```

- The Student Model: [ResNet50_vd](https://arxiv.org/pdf/1812.01187.pdf)(that is ResNet-D in paper). Train student on gpu 0.

``` bash
python -m paddle.distributed.launch --selected_gpus 0 \
  ./train_with_fleet.py \
  --model=ResNet50_vd \
  --data_dir=./ImageNet \
  --use_distill_service=True \
  --distill_teachers=127.0.0.1:9898
```

- To run distillation on clusters, please reference [Run EDL distillation training](./example/distill/README.md)

- Performance benchmark on industrial cluster

| mode | teacher resource | student resource | total batch size | acc1 | acc5 | speed(img/s) |
| :----: | :-----: | :----: | :----: | :----: | :----: | :----: |
| pure train                       | None     | 8 * v100 | 256 | 77.1 | 93.5 | 1828 |
| teacher stduent on the same gpus | 8 * v100 | 8 * v100 | 256 | 79.0 | 94.3 | 656 |
| EDL service distill              | 40 * P4  | 8 * v100 | 256 | 79.0 | 94.5 | 1514 |

<h3 align="center">About Knowledge Distillation in EDL</h3>

- Theory: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
    - Knowledge distillation consists of two parts in general, i.e. strong teachers and weak students. 
    - Student model learns from a teacher or mixture-of-teachers model's feed-forward results to achieve better results.
- Application scenarios of EDL knowledge distillation
    - Teacher models and student models are running on the same GPU devices that training throughputs are not maximized
    - Offline GPU cluster has limited resources but some online GPU resources can be used during training.
    - Heterogenous teacher models can improve student model's performance but are hard to deploy on a single GPU card due to memory limitation.
    - Computation burden of teacher models and student models is hard to balance to maximize the training throughputs.
- Solution:     
    - Deploy teacher models as online inference service through [Paddle Serving](https://github.com/PaddlePaddle/Serving)
    - Online inference services are elastic and are registered to EDL service management modules.
    - Dynamical adaptation of teacher models' online instance to maximize students' training throughputs and resource utilization.

<p align="center">
    <img src="doc/distill.gif" width="550">
</p>

# EDL Framework
## Quickstart:EDL Resnet50 experiments on a single machine in docker:

1. Start a JobServer on one node which generates changing scripts.
 
```
cd example/demo/collective
node_ips="127.0.0.1"
python -u paddle_edl.demo.collective.job_server_demo \
    --node_ips ${node_ips} \
    --pod_num_of_node 8 \
    --time_interval_to_change 900 \
    --gpu_num_of_node 8
```

1. Start a Jobclient which controls the worker process.

```
# set the ImageNet data path
export PADDLE_EDL_IMAGENET_PATH=<your path>
# set the checkpoint path
export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>

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

1. Experiments result on 2 nodes cluster
 
| model| dataset | gpu cards | total batch size | acc1 | acc5 |
| :-----: | ----: | ----: | ----: | ----: | ----: |
| Resnet50 | ImageNet | 16 * v100 | 1024 | 75.5 | 92.8 |

The whole example is [here](example/demo/collective)


## FAQ

TBD

## License

EDL is provided under the [Apache-2.0 license](LICENSE).
