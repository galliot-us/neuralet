[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# TensorRT Engine Generation from a Custom SSD-MobileNet Model 


This repo contains instructions on how to use Neuralet's `l4t-tensorrt-conversion` Docker container. This Docker container installs all the required prerequisites on your [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) and generates a [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) engine from a frozen graph.


## Getting Started

You can read the [Deploying a Custom SSD-MobileNet-V2 Model on the NVIDIA Jetson Nano]() tutorial on our website for detailed instruction on how to generate a TensorRT engine from a frozen graph using the UFF Parser. The following guide will help you get started. 

### Prerequisites

Make sure you have all the prerequisites before continuing to the next section.

**Hardware**
  * NVIDIA Jetson Nano

**Software**
  * [JetPack 4.3](https://developer.nvidia.com/jetpack-4_3_DP)

### Install

After installing the prerequisites, clone this repository to your local system by running this command:

```
git clone https://github.com/neuralet/neuralet.git
cd neuralet/training/tensorrt_generation/ssd/
```

#### Required Files

You should pass a Frozen Inference Graph file to the script through a configuration file from `configs` directory. 
The script is called with `config_ssd_mobilenet_v2_pedestrian.ini` config file by default. It automatically downloads provided retrained pedestrian_ssd_mobilenet_v2 frozen inference graph  from [Neuralet-Models](https://github.com/neuralet/neuralet-models/blob/master/amd64/ped_ssd_mobilenet_v2/frozen_inference_graph.pb) repository which is trained on the [Oxford Town Center](https://megapixels.cc/oxford_town_centre/) dataset.
You can also use other provided config files for `SSD-MOBILENET-V1-COCO` and `SSD-MOBILENET-V2-COCO` or your customized one by changing `CMD` command in Dockerfile to run the script with other configurations.

## Run on Jetson Nano

The provided Docker container runs `build_engine.py` script with the configurations read from the `config_ssd_mobilenet_v2_pedestrian.ini` file. This script generates a TensorRT Engine using the UFF Parser from the frozen graph specified in the config file. The generated engine can be used for the [Smart Social Distancing](https://github.com/neuralet/neuralet/tree/master/applications/smart-distancing) application. 

```
cd neuralet/training/tensorrt_generation/ssd/

# 1) Build Docker image (this step is optional, you can skip it if you want to pull the container from Neuralet's Docker Hub)
docker build -f Dockerfile -t "neuralet/jetson-nano:tf-ssd-to-trt" .

# 2) Run Docker container:
docker run -it --runtime nvidia --privileged --network host -v /PATH_TO_DOCKERFILE_DIRECTORY/:/repo neuralet/jetson-nano:tf-ssd-to-trt
```

### Configurations

You can read and modify the configurations in the `config_ssd_mobilenet_v2_pedestrian.ini` file. Under the `[MODEL]` section, you can customize the model specs, such as model name, model path, the number of classes, objects min size, max size, and input image dimensions.
The `InputOrder` parameter determines the order of `loc_data`, `conf_data`, and `priorbox_data` of the model, which is set equal to the “NMS” node input order in the `.pbtxt` file.

## References
1. [Tensorrt_demos/ssd](https://github.com/jkjung-avt/tensorrt_demos/tree/master/ssd)
2. [TensorRT SampleUffSSD](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleUffSSD)
