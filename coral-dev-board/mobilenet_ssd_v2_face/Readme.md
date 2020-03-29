# Coral MobileNet-SSD V2 (face detection) Object Detection Model:
This is a dockerized implementation of the [Coral](https://coral.ai/) MobileNet-SSD V2 (face detection) for google's [Coral Dev Board](https://coral.ai/products/dev-board/) devices. [Link to the original model.](https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite)

[Single Shot MultiBox Detector (by C. Szegedy et al.)](https://arxiv.org/abs/1512.02325) was released in 2016 with state-of-the-art performance and precision for object detection tasks at 59 frames per second on datasets such as PascalVOC and COCO. The feature extractor of this SSD model is MobileNet V2.

This detection model is trained on [Open Images V4](https://storage.googleapis.com/openimages/web/download_v4.html) dataset and can localize human faces. The input image size of the network must be ```320x320```.

# Codebase architecture:
The ```server-example.py``` is responsible to run inference on the input image and put the result in the output queue. The server will automatically download the edge-TPU compiled tflite model from [neuralet](https://github.com/neuralet/neuralet-models) if it doesn't exist under ```data/models```. The server will start automatically when the container starts.

The ```client.py``` contains a simple script that allows user to prepare their input data and push it to the queue. The input of the server should be a RGB image with the shape of ```(320,320,3)```.

# Getting started:
There are two main ways to run this container. You can build the container from the Dockerfile or pull it from the Dockerhub.
## Build container from source:

```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet

# 2- Build the container
MODEL_NAME=mobilenet_ssd_v2_face
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
MODEL_NAME=mobilenet_ssd_v2_face
cd neuralet/coral-dev-board/$MODEL_NAME
docker run -it --privileged --net=host -v $(pwd)/../../:/repo neuralet/coral-dev-board:$MODEL_NAME

# 3- Run inference on a test image
python3 src/client.py [PATH-TO-IMAGE]

# 4- Terminate the server and stop the container
python3 src/client.py stop
```
