# ERNIE distillation
We show how to distill knowledge from ERNIE to a mini model: BOW and other models on Chinese sentiment task.

## Quick start
### Download dataset
```
wget https://paddle-edl.bj.bcebos.com/distillation/chnsenticorp/data.tgz
tar -xzvf data.tgz
```

### Get the teacher model
```
nohup python -u ./fine_tune.py > finetune.log 2>&1 &
```

When the job completes, the directories needed for distillation: `ernie_senti_server` and `ernie_senti_client` will be generated.

### Or download the teacher model directly
You can also download the teacher model directly and then you needn't generate the model yourself.

```
wget https://paddle-edl.bj.bcebos.com/distillation/chnsenticorp/ernie_senti.tgz
tar -xzvf ernie_senti.tgz
```

### Start a local teacher
```
nohup python -m paddle_serving_server_gpu.serve \
      --model ./ernie_senti_server/ \
      --port 19290 \
      --thread 8 \
      --mem_optim \
      --gpu_ids 0 > teatcher.log 2>&1 &
```

### Start a student
Now the student is BOW. CNN, LSTM, tiny ernie will be added later.

```
python -u distill.py --fixed_teacher 127.0.0.1:19290
```

### Result
| model | dev dataset(acc) | test dataset(acc) |
| :----: | :-----: | :----: |
| BOW     |  0.901    | 0.908 |
| BOW + distillation | 0.905 | 0.915 |
