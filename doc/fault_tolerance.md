# Fault tolerance for sync training
## Design
In the process of training, we may meet that one or more trainers crash. We use checkpoints to continue training.  

There may be several design-tricks for it:

1. How does Paddle save checkpoint itself?  
Paddle implements `save_persistables` to save all persistable variables.

2. How to save user's Python frontend logic?  
Such as current epoch number, step number in an epoch, and the data slice and offset and so on.

3. How to save checkpoints?
  - Which trainer saves the checkpoint?  
    If there are many trainers, the trainer who `rank`==0 will do it.
    
  - Where do we save the checkpoint?  
    It can be saved to the local file system, but eventually, it should be saved to a file-system that can be seen by all trainers such as a distributed HDFS.
    
  - How to guarantee the checkpoint's integrity and correctness?  
    It's a process to save a file and it's not an atomic action but `rm` `rename` `mv` and others should be.
    We can use it and don't change any checkpoint when it's written with a version number. All checkpoints will be saved to the file system with an increment version number. The interface generates a temporay checkpoint file and then `rename` it to valid when it has done.
    
  - when is the checkpoint saved?
    Now the trainer saves checkpoint every epoch and it need not save the data offset, it's very simple. Of course, this method is not friendly when an epoch takes a too long time. We will implement a step level(time-limited) checkpoint interface the next version.
    
## Interface
There are two interfaces `save_check_point` and `load_check_point` to save/load a checkpoint.
There are two arguments should be careful:

1. fs:  
It's an abstract interface to file system and there are two implementations: local file system and HDFS.
You can implement the member function of this class to use the checkpoint interface.

2. train_status:  
Now there is only one member variable `epoch_no` and there will be more variables here after 0.2 version.

## Example
1.save_check_point:

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

2.load_check_point:

```
if args.checkpoint is not None:
    tmp_s = fleet.load_check_point(exe, args.checkpoint, fs=fs, trainer_id=trainer_id)
    if tmp_s is not None:
        train_status = tmp_s
        
for pass_id in range(train_status.next(), params["num_epochs"]):
    train()
```

# Async training
TBD
