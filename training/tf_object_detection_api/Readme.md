# Docker Containers for Training and Deploying Models to Edge Devices

For now there are two docker file in this directory:
## TOCO Converter Tool
`toco-Dockerfile` can quantize and convert a frozen graph to a tflite model with TOCO tools. This container is especially useful for models trained with TensorFlow Object detection API with just specifying the path to the exported frozen graph with following commands:
```
docker build -f tools-toco.Dockerfile -t "neuralet/tools-toco" ./
docker run -v [PATH_TO_FROZEN_GRAPH_DIRECTORY]:/model_dir neuralet/tools-toco --graph_def_file=[frozen graph file]
```
these commands will create a `detect.tflite` file in the graph def directory you mounted to the docker.
you can also specify other parameters like:
`--input_shapes [default is 1,300,300,3`]
`--inference_type [default is QUANTIZED_UINT8]`

## TensorFlow Object Detection API
`tensorflow-od-api-Dockerfile` will install the TensorFlow Object Detection API and its dependecies in the `/models/research/object_detection` directory. For instructions to train an object detection model visit [API Github Repo](https://github.com/tensorflow/models/tree/master/research/object_detection).
Run followings to use this container with CPU support:
```
docker build -f tools-tf-object-detection-api-training.Dockerfile -t "neuralet/tools-tf-object-detection-api-training" ./
docker run -it -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tools-tf-object-detection-api-training
```
Run followings to use this container with GPU support:
```
docker build -f tools-tf-object-detection-api-training.Dockerfile -t "neuralet/tools-tf-object-detection-api-training" ./
docker run -it --gpus all -v [PATH TO EXPERIMENT DIRECTORY]:/work neuralet/tools-tf-object-detection-api-training
```

note that you should install [Nvidia Docker Toolkit](https://github.com/NVIDIA/nvidia-docker) for gpu support.

