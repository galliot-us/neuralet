# Coral Inception V2 model:
This is a dockerized implementation of the [Coral](https://coral.ai/) Inception V2 for google edge-TPU device. [link to model](https://github.com/google-coral/edgetpu/raw/master/test_data/inception_v2_224_quant_edgetpu.tflite)

Inception v2 and Inception v3 were presented in the [same paper](https://arxiv.org/abs/1512.00567). The authors proposed a number of improvements to increase the accuracy and computational efficiency of the models over the [previous version](https://arxiv.org/abs/1409.4842).

This classification model is trained on [ImageNet](http://www.image-net.org/) dataset and can recognize 1000 different object categories. The input image size of the network must be ```224x224```.

# Codebase architecture:
In the ```src/``` directory there are two files: ```server-example.py``` which is responsible to run inference on the input image and put the result in the output queue. The server will automatically download the edge-TPU compiled tflite model file upon start if it doesn't exist. The server will start automatically when the container starts.

The ```client.py``` contains a simple script that allows user to prepare their input data and push it to the queue. The input of the server should be a RGB image with the shape of ```(224,224,3)```.

# Getting started:
There are two main ways to run this container. You can build the container from the Dockerfile or pull it from the Dockerhub.
## Build container from source:

```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet

# 2- Build the container
MODEL_NAME=inception_v2_224_quant_20181026
cd neuralet/coral-dev-board/$MODEL_NAME
docker build -t "neuralet/coral-dev-board:$MODEL_NAME" .

# 3- Run the container
docker run -it --privileged --net=host -v $(pwd)/../../:/repo neuralet/coral-dev-board:$MODEL_NAME

# 4- Run inference on a test image
python3 src/client.py [PATH-TO-IMAGE]

# 5- Terminate the server and stop the container
python3 src/client.py stop
```

## Pull container from Dockerhub:

```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet

# 2- Run the container
MODEL_NAME=inception_v2_224_quant_20181026
cd neuralet/coral-dev-board/$MODEL_NAME
docker run -it --privileged --net=host -v $(pwd)/../../:/repo neuralet/coral-dev-board:$MODEL_NAME

# 3- Run inference on a test image
python3 src/client.py [PATH-TO-IMAGE]

# 4- Terminate the server and stop the container
python3 src/client.py stop
```
