# 服务型蒸馏训练示例
该代码示例修改自[paddle book 数字识别](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits),
示例主要展示服务型蒸馏训练的整个流程。

## 1. 服务型蒸馏训练简介
蒸馏训练分为teacher和student两部分，一般会将teacher和student放在同一张卡上。
服务型蒸馏则将teacher和student分开，将teacher当做inference服务，student除了需要从teacher获取预测结果，其余和普通训练一致。

## 2. 本地训练调试
服务型训练的整个流程主要可分为两大部分：
- teacher模型的获取与部署
- student模型定义与服务的获取

本节先介绍在已有teacher模型和student训练代码下的本地训练调试。训练启动脚本见[run.sh](./run.sh)。
### 2.1 启动本地teacher服务
teacher服务使用paddle_serving部署(serving使用详细请参考[PaddleServing](https://github.com/PaddlePaddle/Serving))。
启动命令如下，其中mnist_model为保存的serving模型，指定线程数为4，服务端口为9292，开启显存优化，使用0号GPU卡。
``` bash
python -m paddle_serving_server_gpu.serve \
  --model mnist_model \
  --thread 4 \
  --port 9292 \
  --mem_optim True \
  --gpu_ids 0
```
### 2.2 运行蒸馏训练
训练启动命令如下，启动服务型蒸馏训练，设置本地固定teacher用于训练调试。
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
python train_with_fleet.py --save_serving_model
```
保存的代码见[train_with_fleet.py](train_with_fleet.py)。模型输入为img，模型输出为prediction，mnist_model为serving模型的目录。
serving_conf为保存的client配置文件。
``` bash
serving_io.save_model("mnist_model", "serving_conf",
                      {img.name: img}, {prediction.name: prediction},
                      test_program)
```
2. 或者在训练完成后，从已保存的paddle inference模型中导出。
``` bash
python train_with_fleet.py
python -c "import paddle_serving_client.io as serving_io; \
serving_io.inference_model_to_serving('recognize_digits_convolutional_neural_network.inference.model', \
    serving_server='mnist_model', serving_client='serving_conf')
```
#### 3.1.2 teacher模型的部署
见2.1 [2.1](#21-teacher)

### 3.2 student模型定义与服务获取
#### 3.2.1 student模型定义
参考[train_with_fleet.py](./train_with_fleet.py)代码。普通训练代码添加服务型蒸馏只需三步，
1. 定义从teacher获取输入的变量，添加至feed列表。
``` python
inputs = [img, label]
soft_label = fluid.data(name='soft_label', shape=[None, 10], dtype='float32')
inputs.append(soft_label)
```
2. 使用蒸馏reader包装原训练reader，返回生成的蒸馏reader
``` python
dr = DistillReader(ins=['img', 'label'], predicts=['prediction'])
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
如果在百度内部，已经在PaddleCloud部署好了整套蒸馏训练流程。可以不用这个部分。

### 4. 部署服务发现服务&Teacher服务注册
TODO。
