# PaddlePaddle EDL: Elastic Deep Learning

While many hardware and software manufacturers are working on
improving the running time of deep learning jobs, EDL optimizes

1. the global utilization of the cluster, and
1. the waiting time of job submitters.

For more about the project EDL, please refer to this [invited blog
post](http://blog.kubernetes.io/2017/12/paddle-paddle-fluid-elastic-learning.html)
on the Kubernetes official blog.

EDL includes two parts:

1. a Kubernetes controller for the elastic scheduling of distributed
   deep learning jobs, and

1. making PaddlePaddle a fault-tolerable deep learning framework.
   This directory contains the Kubernetes controller.  For more
   information about fault-tolerance, please refer to the
   [design](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/v2/design/cluster_train).

We deployed EDL on a real Kubernetes cluster, dlnel.com, opened for
graduate students of Tsinghua University.  The performance test report
of EDL on this cluster is
[here](https://github.com/PaddlePaddle/cloud/blob/develop/doc/edl/experiment/README.md).

## Tutorials

- [Usage](./doc/usage.md)
- [How to Build EDL Component](./doc/build.md)

## Design Docs

- [Fault-Tolerant Training in PaddlePaddle](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/v2/design/cluster_train).
- [Elastic Deep Learning Design Doc](./doc/edl_design_doc.md).

## Future

- Resource Adjustments by EDL
- Support Full-Tolerant Distributed Training in PadldePaddle Fluid.

## FAQ

TBD

## License

PaddlePaddle EDL is provided under the [Apache-2.0 license](LICENSE).