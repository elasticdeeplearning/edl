# 前言
在单节点或者多个节点（物理机器或者虚拟机或者Docker之类的）搭建EDL主要是为开发者准备的：没有集群的情况下也可以对Paddle(计算引擎)模拟进行EDL的测试。
当然，这个过程也有点意思，看着训练进程起起伏伏而且不影响最后的结果，还是蛮有意思的。
Have fun!

# 安装EDL
1. 你可以从源代码编译安装

```
git clone https://github.com/PaddlePaddle/edl
cd edl
mkdir build & cd build
cmake ..
pip install ./python/dist/paddle_edl-0.0.0-cp27-cp27mu-linux_x86_64.whl
```

2. 也可以直接使用`pip`安装我们发布的版本`pip install paddle-edl`

# demo搭建步骤：以单节点为例
1. 我们需要在一个节点上启动JobServer的demo，用来记录训练任务的Pod信息。

```
git clone https://github.com/PaddlePaddle/edl
cd python/paddle_edl/demo/collective
./start_job_server.sh
```
2. 我们需要在(各个)节点上启动一个JobClient的demo，用来管理训练的Pod进程。  

```
mkdir -p resnet50_pod
./start_job_client.sh
```

`resnet50`目录下有一个`package.sh`是用来打包训练脚本和数据的，这里说明一下：  
 2.1 指定ImageNet的数据目录路径  
  `export PADDLE_EDL_IMAGENET_PATH=<your path>`   
  
 2.2 指定`checkpoint`的目录，用来保存checkpoint   
 `export PADDLE_EDL_FLEET_CHECKPOINT_PATH=<your path>`
