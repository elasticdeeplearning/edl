# Introduction
Distilling the Knowledge in a Neural Network[<sup>1</sup>](#r_1) is a different type of training used to transfer the knowledge from the cumbersome models(teachers) to a small model(student) that is more suitable for deployment.

EDL distillation is a large scale and universal solution for knowledge distillation. 

- Decouple the teacher and student models
  - They can run in the same or different nodes and transfer knowledge via network even on heterogeneous machines.            
     Use Distillation on resnet50 as an example: The teachers(Resnet101 for example) can be deployed on P4 GPU cards since they compute forward network generally and the student can be deployed on v100 GPU cards since they need more GPU memory.   

- It's flexible and efficient.
  - Teachers and students can be adjusted elastically in training by the resource utilization  
- Easier to use and deploy.
  - Few lines need to change.
  - End to end use. We release the Kubernetes' deployment solution for you. 

# Design
## Architecture
## Student
## Teacher
## Reader
## Balancer

## Reference
1. <div id="r_1">[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)</div>