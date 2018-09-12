# RICNN_RepeatGongCheng-sPaper
 This project which contain CNNs of paper is from "Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images", it is peoposed in CVPR 2015, The RICNN extract and learn the rotation-invariant feature.
## Usage:
_python RICNN.py_

And you must set the training dataset path and testing dataset path in RICNN.py at first. 
## Note:
In here, I set the tensor of input is (224,224,1), so you must reset the model if you want to use color image dataset.

And H5 is used as dataset reading type, its type is (numbers,224,224,channels)

## Accuracy of RICNN:
We use rotation-mnist-12k dataset to feed for testing accuracy of RICNN, and the accuracy is 98.03%

Name of dataset:rot-mnist-12K
So you should transform the image size in dataset before the run the network.

##This network work on Python 2.7 and Tensorflow 1.6.0
