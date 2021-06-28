# EDL Live Fault Tolerant System Design

## Background

### User Scenarios

In order to meet the needs of ultra-large-scale training such as NLP, INF (Kong Ming) is building a single job to support training clusters of more than 1,000 calories. As the scale of training nodes increases, the probability of job failure due to machine failure also becomes exponential. Increased, because the current Paddle training does not support elastic fault tolerance of hot recovery, the job failure requires reallocation of computing resources to restart training, which leads to low GPU computing resource utilization (or even resource waste), thus improving Paddle fault tolerance and reducing machine User job failures caused by faults and demands for improving GPU resource utilization have gradually become stronger.

INF (Kong Ming), as the operator of computing resources, is its goal to improve the utilization of computing resources. When performing defragmentation operations in order to reduce the fragmentation of computing resources, it is also hoped that Paddle jobs can be migrated and continue to run fault-tolerant.

Overview of Paddle fault tolerance application scenarios:

 - If the node is unreliable (unstable), but you want to not interrupt the user's training process in the event of a node failure
 - When the cluster needs to be defragmented, tasks on a node are required to have the ability to be migrated (hot recovery)
 - The training task runs on a machine that may be preempted by high-priority tasks, and a node may be recycled at any time
 - During the training process, dynamically use more idle resources to accelerate training


### Overall Solution

Paddle distributed training supports node-level (Node-level) thermal recovery and fault tolerance. When one or more of the training nodes fails, the training task has the ability to continue training without interruption (without releasing the overall computing resources) , Including three levels:

1) Thermal recovery and fault tolerance: non-faulty nodes do not need to reallocate computing resources, only faulty nodes redistribute computing resources, and non-faulty nodes wait for the faulty node to recover within the timeout period. If it cannot be recovered within the timeout period, the task fails.
2) Fault tolerance under reduced capacity conditions: the faulty node (or faulty GPU card) will be automatically discarded, and the non-faulty node will continue to train (requires users to dynamically adjust batch_size and learning_rate)
3) Fully elastic fault tolerance: supports dynamic expansion and shrinkage of computing nodes (node and card level)

The fault tolerance solution supports both PS mode and Collective mode. Priority is given to supporting the first level of "hot recovery fault tolerance", and the second and third levels of fault tolerance are gradually being considered


### End Users

1) Distributed tasks that need to be fault-tolerant, preferentially meet the needs of NLP ultra-large-scale parallel training scenarios
2) Computing resource operators who need to defragment to improve computing resource utilization, such as Kong Ming

### Benefits

For an NLP task with 384 cards (48 machines * 8 cards), when the computing resources are sufficient, the time cost of cold recovery and restart is about 1 hour, and the time required for hot recovery is less than 5 minutes.


### Constraints

User interface: additional code is needed to make elastic take effect
Data distribution: Relying on the user's data has been shuffled by itself, no additional data distribution function is provided
Training effect: Elastic fault tolerance requires users to adjust batch size or learning rate by themselves to reduce the impact on convergence and accuracy
mode:

  - Collective mode: Give priority to data parallel mode and dynamic graph support
  - PS mode: First consider the worker node-level fault tolerance of Dense training, then consider PServer fault tolerance, and Sparse training (pslib)

Fault tolerance level: first support for fault hot recovery, then support for fault tolerance for shrinkage, and finally for fault tolerance for fully elastic shrinkage and expansion


## Features

- Meet the fault tolerance requirements of large-scale distributed training of kilocalories in the factory, and support the hot recovery of failed nodes, so that some nodes can be migrated without interrupting the training.
- Meet the current business demands in the factory, and support resource scheduling and operation requirements such as INF IaaS (Kong Ming) layer resource defragmentation and machine mobilization.


## Solutions
### live fault tolerant

Task failure caused by non-user reasons often requires the user to manually re-propose. The task manually re-proposed by the user can only be run from the beginning, which also leads to the invalid use of GPU (or invalid training), in order to reduce the task failure caused by non-user reasons (For example, the failure caused by the kill caused by the overrun, the task failure caused by the machine failure, etc.) At the same time, the GPU usage efficiency is improved. In the case of non-user reasons, the task can support the ability to continue running at a breakpoint.

The goal is a large-scale distributed training task. When one or several training nodes fail, the training task will not be interrupted (without releasing the overall computing resources, due to the need to reinitialize the communication link, all processes will be hot) Restart) have the fault tolerance ability to continue training.


### elastic fault tolerant

Elastic fault tolerance means that the number of collaborative processes can be increased or decreased during the training process. And when the number of processes changes, the training task will not be interrupted.
There are many benefits to using elastic training on a distributed cluster:

1. Fault-tolerant: When some nodes in the task fail, the task can continue to execute, which improves the success rate of the training task.
2. Scalable: When resources are tight, a small number of processes are used to start tasks, thereby quickly starting model iteration; when resources are abundant, processes can be automatically expanded to speed up training.
3. Improve cluster utilization: Using priority scheduling, you can make full use of the idle resources of the shared cluster resources.

In fact, the real realization of fault-tolerant, elastically scalable distributed training is a very complex project, such as

 - When the worker list changes, how to redistribute the unprocessed data under the current epoch;
 - When the number of workers changes, how to adjust the batch size or learning rate so that the convergence of the entire training is not affected too much, etc.

Here we define elastic fault tolerance into three levels:
1) Hot recovery fault tolerance (not support elastic scaling): When a machine or card failure of a distributed job is encountered, a new node will be reassigned, and the training will continue after waiting for the new node to recover. If the new node is restored within the timeout period If the node cannot join the training, the job will fail
2) Fault tolerance under shrinking conditions (supports elastic shrinkage, but does not support elastic expansion)
To

 - If the node fails, discard the failed node and continue training
 - If a single GPU card fails, discard the single failed card and continue training
 
3) Complete elastic fault tolerance (support node and card-level elastic expansion and shrinkage)
 
 - Support on-demand dynamic thermal expansion: expansion without interrupting training operations
 - Support on-demand dynamic thermal shrinkage (including Node-level and GPU card-level thermal shrinkage): shrink without interrupting training operations

The first level is a complete thermal recovery fault-tolerant program, which has a relatively high maturity; the second level has a certain impact on training accuracy and convergence; the third level is an ideal solution for flexibility, which is not yet mature enough.

Prioritize support for hot recovery fault tolerance, and gradually consider elastic fault tolerance.


# Design and Implementation

## Tradeoffs

### Core Design

The solution has two levels of interfaces to the outside:

- One is the interface provided to users, such as the API used in the user networking code, and the command to submit training tasks
API for Paddle: user networking code, start command

- One is the interface provided to the dispatching system or the automatic sensing dispatching system, such as the active expansion interface provided to k8s or the interface of the automatic sensing dispatching system.
API for Scheduler: the interface provided to the scheduling system, the interface of the automatic perception scheduling system

2. Example of use
**Networking code (user interface)**

```
class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """
    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: nn.Layer,
    optimizer,  # SGD
) -> State:
    state = State(arch, model, optimizer)
    return state

def save_checkpoint(state: State, is_best: bool, filename: str):
    """
 
    """

def adjust_learning_rate(optimizer, epoch: int, lr: float) -> None:
    """

    """
    optimizer.set_lr(learning_rate)

def train_resnet():
    fleet.init(is_collective=True)
	
	train_loader = EDLDistributedSampler(train_dataset)

    state = load_checkpoint(args.checkpoint_path)

    for epoch in range(state.epoch, state.total_num_epochs):
	    train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr)

        for batch_id, data in enumerate(train_loader):
            backward()
            optimizer.minimize(avg_loss)

        state.epoch += 1
        save_checkpoint(state)
```

**Start Command**

``` shell
python -m paddl.distributed.elastic.launch
--server_nnodes=MIN_SIZE:MAX_SIZE
--nnodes=2
--nproc_per_node=$NUM_TRAINERS
--rdzv_id=$JOB_ID
--rdzv_backend=etcd
--rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
``` 

### Implementation

The first step: first meet the function of thermal recovery and fault tolerance

 - Introduce Master and ETCD to achieve thermal recovery capacity (not support elastic scaling at the moment)
 - Master single point and ETCD single point (ETCD cluster is preferred)
 - Collective mode: only supports data parallel mode, dynamic and static graphs, and supports hot recovery from failures (not yet support elastic scaling)
 - PServer mode: Support Dense model, static graph, and synchronous training first to meet the trainer's thermal recovery capacity (not support elastic scaling, Pserver does not support fault tolerance)
 - Automatically detect node and card failures, and temporarily not provide external expansion and shrinkage interfaces
 - Add elastic directory to Paddle/python/paddle/distributed/ under Paddle repo, develop Launcher, Master, ETCD Server and other functions
 - Kongming scheduling supports fault-tolerant types

Step 2: Support part of the elastic fault tolerance function

 - PServer mode first supports Dense model, fault tolerance under worker shrinkage conditions (supports worker elastic shrinkage)
 - Collective mode elastic fault tolerance: support fault hot recovery, support node and Worker (card) level shrinkage, support model parallelism
 - PS mode elastic fault tolerance: support Pserver and trainer fault thermal recovery, support Pserver and trainer node-level shrinkage, support synchronous and asynchronous training, Sparse training
 - Master supports multiple copies
 - Paddle-Operator supports fault tolerance types

Step 3: Support fully flexible fault tolerance

 - Support elastic expansion and shrinkage (node ​​and card level)
 - Explore data distribution services
 - Provide external capacity expansion and contraction interfaces, Master independently decides expansion and contraction (need to call the scale interface of the external scheduling system)


### Design Details

#### Core

Main Feature:

1. Support card-level and Node-level hot recovery fault tolerance and elastic fault tolerance
2. Introduce the two roles of Master and ETCD at the same time
3. The Master itself is a process started by Launch. Each node will start a Master process, but only one Master will take effect at the same time (multiple copies, automatic master selection)
4. The ETCD service is hosted and maintained externally. EDL only needs to perceive the host and port of Etcd. ETCD can be in cluster mode or job mode (one etcd per job)
5. The Master provides an API interface, which can be called from the outside for active scaling
6. The Master can actively call the scaling interface of the peripheral scheduling system according to the operating conditions, such as calling the scale interface of k8s
7. Support three user-defined fault tolerance levels

    - Hot recovery capacity (not support elastic scaling): When a machine or card failure of a distributed job is encountered, a new node can be reassigned
    - Fault tolerance under shrinkage conditions (supports elastic shrinkage):
    To
    - If the node fails, discard the failed node and continue training
    - If a single GPU card fails, discard the single failed card and continue training
    To
    - Fully elastic fault tolerance

8. Support PS mode and Collective mode at the same time

The following describes the fault tolerance of PS and Collective modes respectively

#### PS mode fault tolerant

(1) Register the Master with ETCD through paddl.distributed.elastic.launch, select the master and start the Master, each task has only one Master at the same time; pull up the Agent on the Node and start the worker (worker is an abstract concept, (Used to encapsulate the process of training tasks, PS mode only has one process per node)
(2) Agent registers Node with ETCD, Agent perceives Master and actively establishes RPC with Master
(3) The agent continuously observes the status information of the task process on the node, and then reports the information to a Master
(4) Each Worker node obtains the file list from the Master before starting training, each time it obtains a file name or a batch of file name lists, and returns the status finished when all file lists are completed (data distribution is not supported in the early stage, and the user relies on the user to shuffle the data. , Only supports the data parallel mode, and will consider supporting the automatic data distribution function in the future)
(5) When the Master senses a worker node failure, it judges whether to wait for the failed node to recover or automatically discard the node according to the fault tolerance level; at the same time, the peripheral scheduling system can also automatically trigger the shrinkage according to whether the job supports fault tolerance
(6) When the number of Worker nodes changes, in order to ensure the effect of training, the Master needs to notify all PServers through the RPC interface to dynamically adjust the learning rate
(7) After a worker node fails, each agent can perceive the changes of other nodes from ETCD. According to the different fault tolerance levels, each agent obtains all other latest worker nodes from waiting and from the Master, and the agent restarts the worker process (asynchronous training). Can you not restart the Worker of the non-faulty node?) and continue running from the previous checkpoint.
(8) Hot restart all Workers (without releasing computing resources), rebuild the communication link, and resume training according to the checkpoint

#### Collective mode fault tolerant

(1) Register the Master with ETCD through paddl.distributed.elastic.launch, select the master and start the Master, each task has only one Master at the same time; pull up the agent (rendezvous, monitor) and start each worker on the node (worker is An abstract concept used to encapsulate the process of training tasks. In Collective mode, one GPU card corresponds to one worker, and one node has multiple workers)
(2) Agent registers Node with ETCD. Agent senses Master and actively establishes RPC with Master. After each worker is started, it also registers itself with Node under ETCD
(3) Worker No. 0 registers the data set file list to the Master
(4) The agent continuously observes the status information of the distributed task process on the node, and then reports the information to a Master
(5) Each Worker node obtains a file list from the Master before starting training, each time it obtains a file name or a batch of file name lists, and returns the status finished when all file lists have been completed (data distribution is not supported in the early stage, and the user relies on the user to shuffle the data. , Only supports the data parallel mode, and will consider supporting the automatic data distribution function in the future)
(6) Use Rendezvous as a distributed synchronization mechanism for peer discovery to synchronize and collect the information of each worker, including the node list, the role of each node worker, etc., and each agent jointly decides the start, end, and recovery of training, etc.
(7) When a node fails or a worker (a card) on the node fails (card-level failures need to be uploaded to the scheduling system to sense), the node is marked as a failure, and the peripheral scheduling system recognizes the node failure and redistributes it The new node (at the same time, the Master can also automatically sense node failure information and actively trigger fault tolerance or elastic scaling), and other non-faulty nodes hold until the failed node is restored (or reassigned) to re-establish a new set of communications, and then restore from the checkpoint point Continue running.
(8) Hot restart all Workers (without releasing computing resources), rebuild the communication link, and resume training according to the checkpoint

# RoadMap

## Live Fault Tolerant

### Features

- Introduce Master and ETCD to achieve thermal recovery capacity (flexible scaling is not currently supported)
- Master single-point and ETCD single-point (ETCD cluster is given priority, this issue of Master does not support multiple copies, there is a single-point problem, when the Master fails, the entire job will fail)
- Collective mode: only supports data parallel mode, dynamic and static graphs, and supports hot recovery from failures (not yet support elastic scaling)
- PServer mode: first support Dense mode, static images, and synchronous training to meet the trainer's thermal recovery capacity (not support elastic scaling, Pserver does not support fault tolerance)
- Automatically detect node and card failures, and temporarily not provide external expansion and shrinkage interfaces
- Add elastic directory to Paddle/python/paddle/distributed/ under Paddle repo, develop Launcher, Master, ETCD Server and other functions
- Kongming scheduling supports fault tolerance (see "Support kubernetes")

## Elastic Fault Tolerant

### Features

- PServer mode supports Dense mode, worker supports fault tolerance under shrinking conditions (supports worker elastic shrinking)
- Collective mode elastic fault tolerance: support fault hot recovery, support node and Worker (card) level shrinkage, support model parallelism
- Master supports multiple copies
- Paddle-Operator supports fault tolerance types
- Supports mixed scheduling: supports mixed deployment of multiple hardware types, and requires INF support

## Elastic Traning

### Features

- Support elastic expansion and shrinkage (node and card level)
- Explore data distribution services
- PS mode elastic fault tolerance: support Pserver and trainer fault thermal recovery, support Pserver and trainer node-level shrinkage, support synchronous and asynchronous training, Sparse training
- Provide external capacity expansion and contraction interfaces, Master independently decides expansion and contraction (need to call the scale interface of the external scheduling system)
- PaddleCloud support: support flexible job types, support training nodes to specify minimum and maximum values
 


