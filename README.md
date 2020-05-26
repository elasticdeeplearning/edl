# EDL: Elastic Deep Learning

<img src="https://github.com/elasticdeeplearning/artwork/blob/master/horizontal/color/edl-horizontal-color.png" width="500" style="display:inline;vertical-align:middle;padding:2%">

EDL is an Elastic Deep Learning framework designed to help deep learning cloud service providers to build cluster cloud services using deep learning framework PaddlePaddle.

EDL includes two parts:

1. A Kubernetes controller for the elastic scheduling of distributed
   deep learning jobs and tools for adjusting manually.

1. Making PaddlePaddle a fault-tolerable deep learning framework with usability API for job management.

EDL is an incubation-stage project of the [LF AI Foundation](https://lfai.foundation).

<img src="https://github.com/lfai/artwork/blob/master/lfai-project-badge/incubation/color/lfai-projectlogos_incubation-color.png"  width="200" style="display:inline;vertical-align:middle;padding:2%">

While many hardware and software manufacturers are working on
improving the running time of deep learning jobs, EDL optimizes

1. the global utilization of the cluster, and
1. the waiting time of job submitters.

## Key Features:
- Efficiency: Provides parallelism strategies to minimize adjustment overheads.
- Consistency: Accuracy verification on multiple models compared those without scaling.
- Flexibility: Any components can be killed or joined at any time.
- Easy to use: Few lines of code need to be added to support EDL.

## Quick start demo: EDL Resnet50 experiments on a single machine:
We highly **recommand** you run it in our docker:  

1. Start a Jobserver on one node.
 
```
docker pull hub.baidubce.com/paddle-edl/paddle_edl:latest-cuda10.0-cudnn7
cd example/demo/collective
./start_job_server.sh
```

2. Start a Jobclient which controls the worker process.

```
#Set the ImageNet data path
export PADDLE_EDL_IMAGENET_PATH=<your path>
#Set the checkpoint path
export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>

mkdir -p resnet50_pod
./start_job_client.sh
```

3. Experiments result
 
| total batch size | acc1 | acc5 |
| :-----: | ----: | ----: |
| 1024 | 76.0 | 75.8 |


## Design Docs
- A scheduler on Kubernetes:
  -  [Scheduler](./doc/edl_design_doc.md)
- EDL framework on PaddlePaddle:
  -  [Fault-Tolerant Training in PaddlePaddle](./doc/fault_tolerance.md)
  -  [EDL framework](./doc/edl_collective_design_doc.md)

## Applications:
- EDL Distillation:
  - [EDL Distillation design](./doc/edl_distill_design_doc.md)
  - [Run EDL distillation training demo on Kubernetes or a single node](./example/distill/README.md)
  - [EDL Distillation performance: Resnet50](./doc/experiment/distill_resnet50.md)
- EDL CTR
  - [EDL CTR training and deployment on Baidu Cloud](./example/ctr/deploy_ctr_on_baidu_cloud_cn.rst)

## FAQ

TBD

## License

EDL is provided under the [Apache-2.0 license](LICENSE).
