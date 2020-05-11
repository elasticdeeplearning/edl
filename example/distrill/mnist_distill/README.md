# 服务型蒸馏训练示例
该服务型蒸馏代码示例起源于[paddle book 数字识别](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits),后修改自[DGC 示例](https://github.com/PaddlePaddle/Fleet/tree/develop/examples/dgc_example)。
本示例主要展示服务型蒸馏训练的整个流程。

## 测试
目前服务开启(个人开发机上，随时可能关闭)下，可直接运行train_student_with_fleet.py测试student蒸馏训练。
``` bash
# 设置python路径可找到distill_reader
export PYTHONPATH=$PWD/../../:$PYTHONPATH
# 运行student，目前退出还有些bug -_-!!
CUDA_VISIBLE_DEVICES=0 python train_student_with_fleet.py
```

## 服务型蒸馏训练流程
蒸馏训练需要的角色分为teacher和student，一般模式下会将teacher和student放在同一张卡上。
服务型蒸馏训练则将teacher和student分开，使用inference服务作为teacher，student除了部分输入数据需要从服务中获取，其余与普通训练一致。
服务型训练的整个流程主要可分为两大部分：
- teacher模型的获取与部署
- student模型定义与服务的获取

### 1. teacher模型的获取与部署
#### 1.1 teacher模型的获取
在本示例中，运行[train_save_teacher.py](./train_save_teacher.py)即可获取teacher的模型。代码中使用paddle_serving_client保存
可在serving部署的模型，serving具体使用请参考[PaddleServing](https://github.com/PaddlePaddle/Serving)。保存模型代码如下：
``` python
    import paddle_serving_client.io as serving_io
    serving_io.save_model("mnist_model", "mnist_client_conf",
                          {"img": img}, {"prediction": prediction}, test_program)
```
其中img是模型的输入，prediction是模型的输出，mnist_model是用于serving部署的文件目录，mnist_client_conf保存用于client的配置文件。
在本示例中，已有保存的mnist_model和mnist_client_conf，可直接用于部署使用。
#### 1.2 teacher模型的部署
可参考[PaddleServing](https://github.com/PaddlePaddle/Serving)。如下命令
``` bash
python -m paddle_serving_server_gpu.serve --model mnist_model --thread 4 --port 9292 --gpu_ids 0,1 --mem_optim true
```
在0号和1号GPU上启动各启了一个服务，0号GPU服务端口为9292，1号GPU服务端口为9293。这样服务就算部署好了。

### 2. student模型定义与服务获取
#### 2.1 student模型定义
参考[train_student_with_fleet.py](./train_student_with_fleet.py)代码。目前已写好，个人服务在就可正常运行。
``` bash
# 设置python路径可找到distill_reader
export PYTHONPATH=$PWD/../../:$PYTHONPATH
# 运行student，目前退出还有些bug -_-!!
CUDA_VISIBLE_DEVICES=0 python train_student_with_fleet.py
```
[train_student_with_fleet.py](./train_student_with_fleet.py)从[train_with_fleet.py](./train_with_fleet.py)修改而来。增加了蒸馏训练相关的代码。
包括reader的劫持，及模型loss的修改。reader劫持如下：
``` python
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    # 如果为服务蒸馏训练，劫持train_reader，使用蒸馏reader
    if args.use_distill_service:
        assert BATCH_SIZE % 8 == 0
        dr = DistillReader('mnist_client_conf/distill_reader.conf',
                           BATCH_SIZE,
                           d_batch_size=8,
                           capacity=4,
                           occupied_capacity=2)
        dr.set_sample_list_generator(train_reader)
        train_reader = dr.distill_reader()

    # ... some code...

    py_train_reader = fluid.io.PyReader(feed_list=inputs, capacity=2, iterable=True)
    if args.use_distill_service:
        # 蒸馏reader使用decorate_batch_generator装饰
        py_train_reader.decorate_batch_generator(train_reader, place)
    else:
        py_train_reader.decorate_sample_list_generator(train_reader, place)
```

模型loss修改如下：
``` python
    inputs = [img, label]
    test_inputs = [img, label]
    if args.use_distill_service:
        soft_label = fluid.data(name='soft_label', shape=[None, 10], dtype='float32')
        # teacher的输出作为student的输入
        inputs.append(soft_label)
        # 使用student预测的prediction和teacher预测的soft_label计算蒸馏loss，loss函数可根据需要修改
        distill_loss = fluid.layers.cross_entropy(input=prediction, label=soft_label, soft_label=True)
        #distill_loss = fluid.layers.mse_loss(input=prediction, label=soft_label)
        distill_loss = fluid.layers.mean(distill_loss)
        # 定义最终的loss如何计算，上面使用了硬label的loss + 蒸馏loss，下面则只使用了蒸馏loss
        #loss = 0.3 * avg_loss + distill_loss
        loss = distill_loss
    else:
        loss = avg_loss
```
#### 2.2 服务获取
在DistillReader的配置文件[distill_reader.conf](./mnist_client_conf/distill_reader.conf)中，定义了两种服务获取的方式。
一种是固定服务的获取，另一种是通过服务发现来动态获取所需的服务。
如果上面启动了端口在9292和9293的两个服务，则可以设置配置文件中mode=fixed，servers=['127.0.0.1:9292', '127.0.0.1:9293']。
如果已经部署了服务发现服务，则可将mode设为discover，通过host，port连接服务发现服务，指定service_name和require_num获取所需的服务节点。

## 服务发现&服务注册
TODO。
