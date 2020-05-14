# 蒸馏reader使用
依赖于paddle-serving-client。
可以参考distill_reader.py中\_\_main\_\_中的代码。只要服务（个人开发机上，随时可能关闭=。=）还在，安装paddle-serving-client后可以直接运行distill_reader.py测试。
``` bash
python -m pip install paddle-serving-client
python distill_reader.py
```

## 导入模块
使用前需要先将distill_reader.py所在的模块导入到PYTHONPATH变量中。如：
``` bash
# 修改成对应路径
export PYTHONPATH=/home/code/DistillServerBalance/distill_reader:$PYTHONPATH
```
然后在python中导入DistillReader
``` python
from distill_reader import DistillReader
```

## 定义DistillReader的config文件
参考test_mnist_distill_reader.conf文件。分为conf，feed，fetch三个配置部分。
### conf配置
| 配置 | 说明 | 值 |
| --- | --- | --- |
| mode | 获取serving服务节点的模式，包括固定节点以及动态服务获取节点模式 | fixed或discover，必选。分别代表固定模式及动态服务发现模式 |
| servers | mode=fixed固定模式下，配置的serving服务节点, discover模式下无效 | ip:port列表，如['ip0:port0', 'ip1:port1',]，fixed下必须，discover无效不必须 |
| host | 服务发现者的host | (str) ip地址，discover模式下必须，fixed模式下无效不必须 |
| port | 服务发现者的port | (int) 端口号，discover模式下必须，fixed同上 |
| service_name | 服务发现所需的服务名称 | (str) 服务名，discover模式下必须, fixed同上 |
| require_num | 所需的服务节点数 | (int) discover模式下必须, fixed同上。 |
| serving_conf_file | paddle-serving-client所需的配置文件 | (str) 必须 |
`注意`: 动态服务发现模式下，返回的serving服务节点可能会在运行中动态变化，同时返回的服务节点数量可能小于所需的服务节点数。

### feed配置
| 配置 | 说明 | 值 |
| --- | --- | --- |
| feed_vars | 训练输入的变量名 | str列表，如['img', 'label',] |
| feed_types | 训练输入变量对应的类型 | str列表，如['float32', 'int64'] |
| feed_shapes | 训练输入变量对应的shape | shape列表，如[(1, 28, 28), (1,)] |
| predict_feed_ids | 蒸馏预测服务所需变量的id号 | int列表，如[0]，代表蒸馏服务只需'img'输入 |
`注意`：feed_vars的顺序要和reader样本中变量的顺序保持一致。蒸馏预测服务所需的变量名、类型、shape要与paddle-serving-client配置文件内的变量`对应`。
feed_shapes中shape维度为1时, (shape0,)在维度后面加上','。
由于在训练中，所需的数据往往包含样本及样本对应的标签，而在预测服务中一般只需要样本即可，所以需要设置predict_feed_ids。

### fetch配置
| 配置 | 说明 | 值 |
| --- | --- | --- |
| fetch_vars | 所需预测服务的输出 | str列表, 如['prediction', ] |
| fetch_types | 所需预测输出的shape | str列表，如['float32', ] |
| fetch_shapes | 所需预测输出的shape | str列表，如[(10, ), ] |
`注意`: 从预测服务fetch的变量名称、类型、shape要与paddle-serving-client配置文件内的变量对应。
fetch_shapes的shape维度为1时，(shape0,)在维度后面加上','。

## 运行
### 启动serving
TODO
### 启动服务发现服务
TODO
### 运行代码
| 参数 | 说明 | 值 |
| --- | --- | --- |
| conf_file | DistillReader的config文件 | str |
| batch_size | DistillReader输出的batch_size | int |
| d_batch_size | DistillReader给serving预测的batch_size | int |
| capacity | 内存池装下batch的容量 | int |
| occupied_capacity | DistillReader的输出使用了内存池里的内存，用完需要返回。该值表示消费者缓存占用输出的数量，防止内存回收后消费者读入脏数据 | int |
定义好所需的配置文件，准备好paddle-serving-client所需的配置文件。按照\_\_main\_\_中代码逻辑即可。
如上表，DistillReader初始化有五个参数, (conf_file, batch_size, d_batch_size, capacity, occupied_capacity)。
`注意`：目前batch_size需要能被d_batch_size整除。d_batch_size如果设置过大，可能会把服务的显存打爆，使服务挂掉。
`TODO`. 查看DataLoader、PyReader的源码，发现都是copy使用的，不存在缓存原输出的情况，对于它们的occupied_capacity可以设置为0。不过为防止用户使用不当可能造成
的读脏数据，是否改成返回copy生成的numpy，这样就不存在读脏数据的可能性了，虽然多了一次copy可能有些性能损失。


``` python
import numpy as np
from distill_reader import DistillReader

# 训练样本，标签reader，yield sample_list
def _reader():
    img = np.array([(i+1)/28.0 for i in range(28)] * 28, dtype=np.float32).reshape((1, 28, 28))
    label = np.array([100], dtype=np.int64)
    for i in range(24):
        yield 8 * [(img, label)]
    yield 2 * [(img, label)]

# 初始化dr，其中batch_size=32，给serving发送数据的batch_size=4，occupied_capacity=2
dr = DistillReader('test_mnist_distill_reader.conf', 32, 4, capacity=4, occupied_capacity=2)

# 设置sample_list 输入reader
dr.set_sample_list_generator(_reader)

# 获取输出distill_reader，该reader会将配置文件中的feed和fetch拼接起来
# 如predict_feed_vars=['img'] fetch_vars=['prediction']
# 则返回封装['img', 'prediction']后的batch。
distill_reader = dr.distill_reader()

for epoch in range(300):
    for step, batch in enumerate(distill_reader()):
        # print('----step={}, predict_shape={}, predict[0]={} ----'.format(step, batch[-1].shape, batch[-1][0]))
        pass
    if epoch % 10 == 0:
        print('^^^^^^^^^^^^^ epoch={} predict[0][0]={}^^^^^^^^^^^^^^'.format(epoch, batch[-1][0][0]))
```
