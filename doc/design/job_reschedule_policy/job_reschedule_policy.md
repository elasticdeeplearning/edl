# edl gpu任务优化重调度设计

## 现状

目前Kubernetes的调度器本质上是通过集群当前状态的信息采集，而不没有太多地考虑到集群自身资源上的变化，例如添加特定资源机器，删除特定资源机器，并且我们也会对每一台机器上的标签进行修改或修正。这样会导致某些现存任务有重新调度的诉求，这些诉求主要体现在：

* 节点的新建和删除后，需要对原有gpu任务的分布进行优化的需求
* 有一些节点上的综合资源使用率过高或者过低，需要平衡的需求
* 节点/pod亲和性标签的变化，需要重新满足的需求


## 目标

* 如同网络策略一样，设计一个重调度策略模型，来发现这些需要重新调度的任务容器，并进行容器的重调度，另外，同调度器相同，每一个策略需要设定相应的weight。

* 在任务框架支持容错和扩缩容功能的情况下，能自动执行reschedule策略来进行重调度，优化集群资源利用。


## 框架策略示例


### 避免单个节点上gpu资源利用过高或过低

```
apiVersion: "rescheduler/v1alpha1"
kind: "ReschedPolicy"
policy:
  "GpuUtilization"
     weight: 10
```

### 避免同一节点上运行同一任务下的不同副本

```
apiVersion: "rescheduler/v1alpha1"
kind: "ReschedPolicy"
policy:
  "NoDuplicates"
     weight: 10
```

### 随node selector改变而重调度对应实例

```
apiVersion: "rescheduler/v1alpha1"
kind: "ReschedPolicy"
policy:
  "NodeAffinity"
     weight: 10
``` 


## 可能遇到的问题

* 和调度一样，反调度的问题也是相对应默认值的确定，各个策略间如何加权平均决定了整体重调度的效率。

## 测试用例

暂无

## 后续计划

暂无


