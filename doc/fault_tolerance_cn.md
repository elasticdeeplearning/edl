# 同步训练的FaultTolerance
## 设计思路
在训练的过程中我们可能会碰到因为各种的问题造成的训练单个（或者多个）trainer挂掉的问题。我们采用checkpoint的方式记录当前状态，保证重启之后训练任务能够正常运行。
这里边可能有几个地方需要考虑：

1. Paddle本身的checkpoint
Paddle本身提供`save_persistables `保存所有持久的变量。

2. 用户python端逻辑的checkpoint问题
主要是当前epoch number，数据切分方法和位置等。

3. checkpoint保存的问题
   - 谁来保存
   如果有多个trainer节点，我们一般会选择rank=0的trainer来负责保存checkpoint

   - 保存的位置
   可以保存到本地，但是最终要保存到重启任务能够看到的文件系统里边，如分布式的HDFS文件系统。

   - 如何确保checkpoint的正确性
   保存文件一个持续性的过程，不是一个原子性的过程，不能保证事务性。但是一般的文件系统的操作`mv` `rename` `rm` 是。
   可以利用这个特点，对已经保存的checkpoint不变，递增当前的 checkpoint的版本号，先写入一个临时文件，完成之后再rename成一个有效文件名的checkpoint。

   - 何时保存
   我们现在推荐的方式是每一个epoch保存一次。因为一个epoch完成之后，可以认为两个epoch数据上没有关系。这样我们只需要保存当前的epoch号就可以了，不用保存当前的文件逻辑切分和位置等。减少了复杂度。当然，这种方式对一个epoch过大的的不友好。我们准备以后的版本开发step级别（时间）的checkpoint

## 接口介绍
Paddle提供`save_check_point`和`load_check_point`两种方式来存、读checkpoint。
其中有两个参数需要注意一下:
1.fs
这个是我们对文件系统的抽象，目前的实现有两种：本地和远程HDFS。您可以实现自己的`FS`类来实现保存和读取checkpoint的功能

2.train_status
目前该类只有`epoch_no`的类变量，0.2以后的版本将尝试增加用户自定义的member等更多的值。

## 使用样例
1. save_check_point的样例:

```
if trainer_id == 0:
    saved_status = TrainStatus(pass_id)
    if args.checkpoint:
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)

        print("save_check_point:{}".format(args.checkpoint))
        fleet.save_check_point(executor=exe, train_status=saved_status,
            path=args.checkpoint, fs=fs)#, main_program=fleet._origin_program)
```

2. load_check_point的样例:

```
if args.checkpoint is not None:
    tmp_s = fleet.load_check_point(exe, args.checkpoint, fs=fs, trainer_id=trainer_id)
    if tmp_s is not None:
        train_status = tmp_s
```


#  异步训练的FaultTolerance
TBD
