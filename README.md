# EDL: Elastic Deep Learning

<img src="https://github.com/elasticdeeplearning/artwork/blob/master/horizontal/color/edl-horizontal-color.png" width="500" style="display:inline;vertical-align:middle;padding:2%">

EDL is an Elastic Deep Learning framework designed to help deep learning cloud service providers to build cluster cloud services using deep learning frameworks such as PaddlePaddle and TensorFlow. EDL includes a Kubernetes controller, PaddlePaddle auto-scaler, which changes the number of processes of distributed jobs to the idle hardware resource in the cluster, and a new fault-tolerable architecture.

EDL is an incubation-stage project of the [LF AI Foundation](https://lfai.foundation).

<img src="https://github.com/lfai/artwork/blob/master/lfai-project-badge/incubation/color/lfai-projectlogos_incubation-color.png"  width="200" style="display:inline;vertical-align:middle;padding:2%">

While many hardware and software manufacturers are working on
improving the running time of deep learning jobs, EDL optimizes

1. the global utilization of the cluster, and
1. the waiting time of job submitters.

For more about the project EDL, please refer to this [invited blog
post](https://kubernetes.io/blog/2017/12/paddle-paddle-fluid-elastic-learning/)
on the Kubernetes official blog.

EDL includes two parts:

1. a Kubernetes controller for the elastic scheduling of distributed
   deep learning jobs, and

1. making PaddlePaddle a fault-tolerable deep learning framework.
   This directory contains the Kubernetes controller.  For more
   information about fault-tolerance, please refer to the
   [design](./doc/fault_tolerance.md).

We deployed EDL on a real Kubernetes cluster, dlnel.com, opened for
graduate students of Tsinghua University.  The performance test report
of EDL on this cluster is
[here](https://github.com/PaddlePaddle/cloud/blob/develop/doc/edl/experiment/README.md).

## Tutorials

- [Usage](./doc/usage.md)
- [How to Build EDL Component](./doc/build.md)
- [Run CTR Training and Deployment on Baidu Cloud](./example/ctr/deploy_ctr_on_baidu_cloud_cn.rst)

## Design Docs
- Collective communication pattern
  -  [Fault-Tolerant Training in PaddlePaddle](./doc/fault_tolerance.md).
  -  [Elastic Deep Learning Design Doc:compute engine](./doc/edl_collective_design_doc.md).
  -  [Elastic Deep Learning Design Doc:Scheduler](./doc/edl_design_doc.md).
  -  [Run Elastic Deep Learning Demo on a sinle node](./doc/collective_demo.md).

## FAQ

TBD

## License

EDL is provided under the [Apache-2.0 license](LICENSE).
