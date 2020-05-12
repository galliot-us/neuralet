# Docker Containers for Training and Deploying Models to Edge Devices With TensorFlow Object Detection API


This is a guide on how to use Neuralet's Docker containers that are designed for training and deploying TensorFlow Object Detection API models to edge devices, such as [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) and [Coral Edge TPU](https://coral.ai/products/). 

To learn more about the quantization of TensorFlow Object Detection API models, read [this tutorial](https://neuralet.com/docs/tutorials/tf-object-detection-api-model-quantization/) on Neuralet's website.

## TOCO Docker Container

This Docker container is especially useful for models that are trained with the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). It can quantize and convert a frozen graph to a `tflite` model using [TOCO](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/toco) tools. You will only need to specify the path to the exported frozen graph to get the `tflite` file.

To run this container, you can either build the Docker container from source or pull the container from Docker Hub.

### Build the Container from Source:

```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet
cd training/tf_object_detection_api

# 2- Build the container
docker build -f toco-Dockerfile -t "neuralet/toco" .

3- Run the container
docker run -v [PATH_TO_FROZEN_GRAPH_DIRECTORY]:/model_dir neuralet/toco --graph_def_file=[frozen graph file]
```

### Pull the Container from Docker Hub:
```
docker run -v [PATH_TO_FROZEN_GRAPH_DIRECTORY]:/model_dir neuralet/toco --graph_def_file=[frozen graph file]
```

Running the container will create a `detect.tflite` file in the graph def directory that you have mounted to the docker.

You can also override other parameters, such as `--input_shapes[default:1,300,300,3]` and 
`--inference_type[default:QUANTIZED_UINT8]`, when running the Docker container.


## TensorFlow Object Detection API Docker Container

The `tensorflow-od-api-Dockerfile` will install the TensorFlow Object Detection API and its dependencies into `/models/research/object_detection` directory. For instructions on how to train an object detection model, visit the [API's GitHub repo](https://github.com/tensorflow/models/tree/master/research/object_detection). To run this Docker container, you should either build the Docker container from source or pull the container from Docker Hub.


1- Run with CPU support:

### Build the Container from Source:
```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet
cd training/tf_object_detection_api

# 2- Build the container
docker build -f tensorflow-od-api-Dockerfile -t "neuralet/tensorflow-od-api" .

3- Run the container
docker run -it -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tensorflow-od-api
```
### Pull the Container from Docker Hub:
```
docker run -it -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tensorflow-od-api

``` 

2- Run with GPU support:


You should have the [Nvidia Docker Toolkit](https://github.com/NVIDIA/nvidia-docker) installed to be able to run the Docker container with GPU support.

### Build the Container from Source:
```
# 1- Clone the repository
git clone https://github.com/neuralet/neuralet
cd training/tf_object_detection_api

# 2- Build the container
docker build -f tensorflow-od-api-Dockerfile -t "neuralet/tensorflow-od-api" .

3- Run the container
docker run -it --gpus all -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tensorflow-od-api
```
### Pull the Container from Docker Hub:
```
docker run -it --gpus all -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tensorflow-od-api
```

