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

For more about the project EDL, please refer to this [invited blog
post](https://kubernetes.io/blog/2017/12/paddle-paddle-fluid-elastic-learning/)
on the Kubernetes official blog.

## Tutorials
- [Run CTR Training and Deployment on Baidu Cloud](./example/ctr/deploy_ctr_on_baidu_cloud_cn.rst)
- [Run EDL distill training demo on Kubernetes or a single node](./example/distill/README.md)
- [Run Elastic Deep Learning Demo on a sinle node](./example/collective/README.md)

## Design Docs
- A scheduler on Kubernetes:
  -  [Scheduler](./doc/edl_design_doc.md)
- EDL framework on PaddlePaddle:
  -  [Fault-Tolerant Training in PaddlePaddle](./doc/fault_tolerance.md)
  -  [EDL framework](./doc/edl_collective_design_doc.md)
  -  [EDL Distillation](./doc/edl_distill_design_doc.md)

## Experiments:

- [Auto-scaling Experiment](https://github.com/PaddlePaddle/cloud/blob/develop/doc/edl/experiment/README.md)
- [Distill training on Resnet50](./doc/experiment/distill_resnet50.md)

## FAQ

TBD

## License

EDL is provided under the [Apache-2.0 license](LICENSE).
