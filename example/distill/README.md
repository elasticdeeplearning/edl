# Purpose
This article illustrates how to run distill demo on Kubernetes cluster or one single machine.

## On a single node
## How to run with dynamic teacher
In addition to the teacher and student, a discovery service and a database is required.
`Once the database and discovery service is deployed, they can be used permanently for different students and teachers.`
1. [Install & deploy redis](https://redis.io/download) as a database.
The teacher service will be registered in the database, and discovery service query teacher from database.
``` bash
redis-server
``` 
2. Deploy distill discovery service. The service also provides a balanced function.
``` bash
python -m paddle_edl.distill.redis.balance_server \
  --db_endpoints REDIS_HOST:REDIS_PORT \
  --server DISCOVERY_IP:DISCOVERY_PORT
```
3. Register teacher to database. You can register or stop teacher any time. 
``` bash
python -m paddle_edl.distill.redis.server_register \
  --db_endpoints REDIS_HOST:REDIS_PORT \
  --service_name TEACHER_SERVICE_NAME \
  --server TEACHER_IP:TEACHER_PORT
```
4. Use `set_dynamic_teacher` get dynamic teacher from discovery service.
``` python 
dr = DistillReader(ins=reader_ins, predicts=teacher_predicts)
dr.set_dynamic_teacher(DISCOVERY_IP:DISCOVERY_PORT, TEACHER_SERVICE_NAME)
train_reader = dr.set_sample_list_generator(train_reader)
``` 
The run student code.
``` python
python train_student.py
```

## On Kubernetes

We have built the docker images for you and you can start a demo on Kubernetes immediately:
TBD
