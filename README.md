# Covid Detection with X-Ray and PyTorch 

## Introduction

We will use the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) to see if we can classify X-Ray images and detect the presence of Covid on them.

The datasets contains auround 3900 images. Among them, 1200 come from COVID-19 positive patients, 1345 from patients suffering viral pneumonia and finally 1341 from healthy people.

We will take ResNet-18 and fine-tune it.

All this will be done using the PyTorch framework.

## The Dataset

The dataset consists of 3886 png images. The resolution is either 1024x1024 and 256x256 pixels. ResNet-18 expects 224x224 pixels images as input, so we will resize them thanks to torchvision transforms.

![X-ray](./COVID_1128.png?raw=true "X-ray of a Covid-19 positive patient")


## Resnet-18   

ResNet were first introduced in 2015 with [this paper](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun from Microsoft Research. 

It's a Deep Residual Network trained on Imagenet that obtained state-of-the-art performance on image recognitions tasks.

## Steps

We will build a dataset class that will, among other things, performs data augmentation thanks to torchvision.transform.

The following steps will be executed:
* resize to 224x224
* apply random horizontal flip
* normalize the data the same way as ImageNet.

We will import the pretrained ResNet model from torchvision. 
ResNet18 was trained on Imagenet, which has 1000 classes. We only have 3, Normal, viral pneumonia and covid-19.

So we will want to change the last fully connected layer and change it to 3 output features (looking at the last layer, we can see it has out_features=1000)

Cross entropy has been chosen as a loss function and Adam as the optimizer.

Finally we will visualize our results.

## Results.

## References

[COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) 
(https://arxiv.org/abs/1512.03385)

