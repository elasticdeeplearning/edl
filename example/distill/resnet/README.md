# ResNeXt101_32x16d_wsl distill ResNet50_vd

## Local test
### start local teacher
start ResNeXt101_32x16d_wsl teacher on gpu 1
``` bash
bash ./scripts/start_local_teacher.sh
```
### train student with local teacher
At another terminal, train resnet50_vd student on gpu 0.
``` bash
bash ./scripts/train_student.sh
```
