# Introduction
Deep learning training under collective communication pattern is widely used on Image classify, NLP and speech because its numerical accuracy is stable and it's easy to reproduce.
This document will describe how to design the EDL function of it.

# Design tricks
When a user upgrades his(her) training program from a single node to multiple nodes, he(she) should add the super parameter: the number of the nodes. For example, the data may need to cut between nodes, the batch size may need to adjust and so on.
And when a user adjusts his training program from a multi-node to EDL, the parameter's consistency should be considered. Parameters on all the nodes should consist of all nodes after the nodes are changed. We should try to reduce the workload of this transform.

There are some tricks of it:

1. How to save the python frontend logic.
Such as how data is cut and other logic outside the Paddle framework. These parameters(logic) are user-defined and we can't hold all of them in the framework itself. So we use the stop-resume method to implement the EDL. The only additional super parameter is the node's number.

2. How to guarantee the accuracy, the result's reproducible and the expansion efficiency.
Before a training job is launched, the user should specify the range of the training nodes' number and also the total batch size: keep the same regardless of whether the trainer nodes are changed or maintain linear increase and decrease. Some models such as Resnet50's learning rate should be adjusted when the total batch size is larger than 8K.
However keep the same total batch size will meet the problem of expansion efficiency: if batch size decreases on a single node, the training performance will decreases.
Consider the above problem, the user should decide his policy of batch size.

3. How to reduce the user program changes of transform.
`stop-resume` method needs `save_checkpoint` and `load_checkpoint` interface and it's hard to hide these actions under the other training interfaces. So the user should call them explicitly.
And there are no additional changes that should be thought about.

4. How to adapt to multiple clusters.
There are still many online or offline cluster systems in production although the Kubernetes are widely used now. To adapt to these, we implement a middle layer: Jobserver. EDL uses it to connect to the multiple clusters.

5. How to prevent meaningless scaling.
  - When a training job is nearing its end, it's meaningless to scale it(scale_in or scale_out). It will decrease the job's efficiency.
  - In most scenes, it's better to scale_out jobs with high resource utilization instead of the lowers.
So Paddle should report its job performance information to the scheduler and then the scheduler can adjust its scheduling strategy.

6. How to split the data between nodes.
Changes in nodes number may need data to adjust segmentation. So a node needs to see all the data. One choice is that download all of them and the other is to mount a distributed file system(such as Ceph).

# Archtecture
<img src="images/edl-arch.png" width="750">

## Launcher module
<img src="images/launcher.png" width="450">
The module `Launcher` is responsible for coordinating multiple nodes.

## Trainer module
<img src="images/trainer.png" width="320">.
The module `Trainer` is responsible for `save_checkpoint` `load_checkpoint` for EDL.
