# 服务型蒸馏训练示例
代码示例修改自[paddle book 数字识别](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits),
示例用于展示服务型蒸馏训练的整个流程。

## 1. 服务型蒸馏训练简介
蒸馏训练包含teacher和student两部分，一般会将teacher和student放在同一张卡上。
服务型蒸馏则将teacher和student分开，将teacher当做inference服务。student发送样本数据到teacher，从服务中获取预测结果用于训练。
edl将student的发送接收数据部分封装成了DistillReader，用户关注这部分如何使用即可。

## 2. 本地训练调试
服务型训练的整个流程主要可分为两大部分：
- teacher模型的获取与部署
- student模型定义与服务的获取

这两部分在后面章节介绍，本节介绍在已有teacher模型和student训练代码下，本地蒸馏训练调试的流程，启动脚本见[run.sh](./run.sh)。
启动前先安装依赖的python包
``` bash
pip install paddle-edl paddle-serving-client paddle-serving-server-gpu
```
### 2.1 启动本地teacher服务
teacher服务使用paddle_serving部署(serving使用详细请参考[PaddleServing](https://github.com/PaddlePaddle/Serving))。
其中teacher模型是mnist训练出的一个cnn模型，输入为{name='img', shape=(1, 28, 28), dtype='float32'}的图像，
输出为{name='fc_0.tmp_2', shape=(10,), dtype='float32'}的图像类别预测概率值。
``` bash
wget --no-check-certificate https://paddle-edl.bj.bcebos.com/distill_teacher_model/mnist_cnn_model.tar.gz
python -m paddle_serving_server_gpu.serve \
  --model mnist_cnn_model \
  --thread 4 \
  --port 9292 \
  --mem_optim True \
  --gpu_ids 0
```
### 2.2 运行蒸馏训练
训练启动命令如下，use_distill_service设置训练为蒸馏训练，distill_teachers设置本地固定teacher地址用于训练。
``` bash
export CUDA_VISIBLE_DEVICES=0 
python train_with_fleet.py --use_distill_service True --distill_teachers 127.0.0.1:9292
```

## 3. 服务型蒸馏训练流程
### 3.1 teacher模型的保存与部署
#### 3.1.1 teacher模型的保存
teacher服务使用paddle_serving部署，需保存成serving模型。可以有两种方式获取(详见[如何保存Serving模型](https://github.com/PaddlePaddle/Serving/blob/develop/doc/SAVE.md))。
1. 直接在训练中保存serving模型。
``` bash
python train_with_fleet.py --save_serving_model True
```
保存的代码见[train_with_fleet.py](train_with_fleet.py)。模型输入为img，模型输出为prediction。
模型保存到output目录，mnist_model为保存的serving模型，serving_conf为保存的client配置文件。
``` bash
serving_io.save_model("output/mnist_cnn_model", "output/serving_conf",
                      {img.name: img}, {prediction.name: prediction},
                      test_program)
```
2. 或者在训练完成后，从已保存的paddle inference模型中导出。
``` python
import paddle_serving_client.io as serving_io
serving_io.inference_model_to_serving('recognize_digits_convolutional_neural_network.inference.model', \
    serving_server='mnist_cnn_model', serving_client='serving_conf')
```
#### 3.1.2 teacher模型的部署
见2.1。

### 3.2 student模型定义与服务获取
#### 3.2.1 student模型定义
参考[train_with_fleet.py](./train_with_fleet.py)代码。普通训练代码添加服务型蒸馏只需三步，
1. 定义从teacher获取输入的变量，添加至feed列表。
``` python
inputs = [img, label]
soft_label = fluid.data(name='soft_label', shape=[None, 10], dtype='float32')
inputs.append(soft_label)
```
2. 定义蒸馏Reader输入及需获取的预测结果。按原DataLoader使用方式，包装训练reader，返回蒸馏训练reader。
返回的结果会将输入预测拼接起来，下面代码的表现为返回(img, label, fc_0.tmp_2)组成的列表。
``` python
dr = DistillReader(ins=['img', 'label'], predicts=['fc_0.tmp_2'])
train_reader = dr.set_sample_list_generator(train_reader)
```
3. 定义蒸馏loss
``` python
# prediction为student预测结果，soft_label为teacher预测结果，计算loss
distill_loss = fluid.layers.cross_entropy(
    input=prediction, label=soft_label, soft_label=True)
```
#### 3.2.2 服务的获取
服务的获取包括两种方式。
1. 固定teacher。
示例中使用了本地启动的固定teacher。通过set_fixed_teacher接口即可设置固定teacher。
``` python
dr.set_fixed_teacher('127.0.0.1:9292')
```
2. 从服务发现中获取动态teacher。
从服务发现中获取可动态扩缩容的teacher。
``` python
# 设置服务发现地址，及需要获取的teacher服务名
dr.set_dynamic_teacher(discovery_servers, teacher_service_name)
```

### 4. 部署服务发现服务&Teacher服务注册
#### 4.1 部署服务发现
1. 依赖于redis数据库，[下载安装redis](https://redis.io/download)。
2. 部署balance服务发现。其中server指定balance服务对外服务的地址，db_endpoints为启动的redis数据库地址。
``` python
python -m paddle_edl.distill.redis.balance_server \
  --server 127.0.0.1:7001 \
  --db_endpoints 127.0.0.1:6379
```
#### 4.2 服务注册
在已启动好teacher后，需要往redis数据库注册teacher服务。
其中db_endpoints为redis数据库地址，server为teacher对应的地址，service_name为teacher注册的服务名。
``` python
python -m paddle_edl.distill.redis.server_register \
  --db_endpoints 127.0.0.1:6379 \
  --server 127.0.0.1:9292 \
  --service_name MnistDistill
```

服务发现部署完成后，使用set_dynamic_teacher('127.0.0.1:7001', 'MnistDistill')接口，student会向发现服务请求
所需的teacher，服务发现服务则会从数据中查询已注册的服务，返回给student。
