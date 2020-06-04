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
A complete example is [here](./mnist_distill/)
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
