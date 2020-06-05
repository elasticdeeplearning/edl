# Purpose
This article illustrates how to run distill demo on Kubernetes cluster or one single machine.

## On Kubernetes

We have built the docker images for you and you can start a demo on Kubernetes immediately:

1. Get the yaml files from: `edl/example/distill/k8s/`
2. Use kubectl to create resources from them, such as `kubectl create -f student.yaml`  

## On a single node
### How to change from a normal train program to an EDL distill train program
1. Define an input to represent the variable obtained from the teacher.
2. Use DistillReader to define the original reader input and the variables that need to be obtained from the teacher.
Then set train_reader as the data source of DistillReader.
3. Use student's prediction and teacher's prediction to define loss function.
``` python
# 1. define teacher prediction
soft_label = fluid.data(name='soft_label', shape=[None, 10], dtype='float32')
inputs.append(soft_label)

# 2. define DistillReader
dr = DistillReader(ins=['img', 'label'], predicts=['fc_0.tmp_2'])
train_reader = dr.set_sample_list_generator(train_reader)

# 3. define distill loss function
distill_loss = fluid.layers.cross_entropy(
    input=prediction, label=soft_label, soft_label=True)
distill_loss = fluid.layers.mean(distill_loss)
loss = distill_loss

# start distill train.
# data includes the original reader input and the prediction results obtained from the teacher,
# that is (img, label, soft_label)
for data in train_reader():
    metrics = exe.run(main_program, feed=data, fetch_list=[loss, acc])
```
### How to run on a single node
#### Run with fixed teacher
**A complete example is [here](./mnist_distill)**
1. First, you need deploy teacher. We using [Paddle Serving](https://github.com/PaddlePaddle/Serving) to deploy teacher.
You can see [here](https://github.com/PaddlePaddle/Serving/blob/develop/doc/SAVE.md) how to save a Paddle Serving model.
``` bash
python -m paddle_serving_server_gpu.serve \
  --model TEACHER_MODEL \
  --port TEACHER_PORT \
  --gpu_ids 0
```
2. Then, define the student code. Use `set_fixed_teacher` to set fixed teacher addr.
``` python
dr = DistillReader(ins=reader_ins, predicts=teacher_predicts)
dr.set_fixed_teacher(TEACHER_IP:TEACHER_PORT)
train_reader = dr.set_sample_list_generator(train_reader)
```
The run student code.
``` python
python train_student.py
```
#### Run with dynamic teacher
In addition to the teacher and student, a discovery service and a database is required.
`Once the database and discovery service is deployed, they can be used permanently for different students and teachers.`
1. [Install & deploy redis](https://redis.io/download) as a database.
The teacher will be registered in the database, and discovery service will query teacher
service from database.
``` bash
redis-server
``` 
2. Deploy distill discovery service. The service also provides a balanced service.
``` bash
python -m paddle_edl.distill.redis.balance_server \
  --db_endpoints REDIS_HOST:REDIS_PORT \
  --server DISCOVERY_IP:DISCOVERY_PORT
```
3. Register teacher to database.
``` bash
python -m paddle_edl.distill.redis.server_register \
  --db_endpoints REDIS_HOST:REDIS_PORT \
  --service_name TEACHER_NAME \
  --server TEACHER_IP:TEACHER_PORT
```
4. Use `set_dynamic_teacher` get teacher from discovery service.
``` python 
dr = DistillReader(ins=reader_ins, predicts=teacher_predicts)
dr.set_dynamic_teacher(DISCOVERY_IP:DISCOVERY_PORT, TEACHER_NAME)
train_reader = dr.set_sample_list_generator(train_reader)
``` 
The run student code.
``` python
python train_student.py
```
