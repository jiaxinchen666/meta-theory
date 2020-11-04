# A Closer Look at the Training Strategy for Modern Meta-Learning

## Enviroment
 - Python3
 - Pytorch 0.4
 - json

## Regression

## LOO training strategy

### Bilevel programming

```python ./bilevel_code/regression_loo.py --n_task [NUMBER OF TRAINING TASKS] --train_shot [NUMBER OF SHOTS] --test_shot [NUMBER OF SHOTS]```

### MAML

```python ./protonet_maml_code/train.py --dataset "regression_loo" --model "ReMaml" --method "re_maml" --n_episodes [NUMBER OF TRAINING TASKS] --n_shot [NUMBER OF SHOTS]```

## S/Q training strategy

### Bilevel Programming

```python ./bilevel_code/regression_sq.py --n_task [NUMBER OF TRAINING TASKS] --train_shot [NUMBER OF SHOTS] --test_shot [NUMBER OF SHOTS] --query [NUMBER OF QUERIES]```

### MAML

```python ./protonet_maml_code/train.py --dataset "regression" --model "ReMaml" --method "re_maml" --n_episodes [NUMBER OF TRAINING TASKS] --n_shot [NUMBER OF SHOTS] --n_query [NUMBER OF QUERIES]```


## Classification

### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

### ProtoNet

```python ./protonet_maml_code/train.py --dataset "miniImagenet" --model "Conv4" --method "protonet" --n_episodes [NUMBER OF TRAINING TASKS] --n_shot [NUMBER OF SHOTS] --n_query [NUMBER OF QUERIES] --gap True```

### MAML

```python ./protonet_maml_code/train.py --dataset "miniImagenet" --model "Conv4" --method "maml" --n_episodes [NUMBER OF TRAINING TASKS] --n_shot [NUMBER OF SHOTS] --n_query [NUMBER OF QUERIES] --gap True```

## Results

```python ./bilevel_code/bilevel_regression_visualization```

```python ./protonet_maml_code/maml_regression_visualization```

## References
This code is built on

* Bilevel programming
https://github.com/cyvius96/prototypical-network-pytorch
* MAML and ProtoNet
https://github.com/wyharveychen/CloserLookFewShot

