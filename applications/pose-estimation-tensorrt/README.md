# TensorRT OpenPifPaf Pose Estimation 

TensorRT OpenPifPaf Pose Estimation is a Jetson-friendly application that runs inference using a [TensorRT](https://developer.nvidia.com/tensorrt) engine to extract human poses. The provided TensorRT engine is generated from an ONNX model exported from [OpenPifPaf](https://github.com/vita-epfl/openpifpaf) version 0.10.0 using [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt) repo. 

You can read [this article]() on our website to learn more about the TensorRT OpenPifPaf Pose Estimation application.

## Getting Started

The following instructions will help you get started.

### Prerequisites

**Hardware**
* [NVIDIA Jetson TX2](https://developer.nvidia.com/embedded/jetson-tx2)

**Software**
* You should have [Docker](https://docs.docker.com/get-docker/) on your device.

### Install


```bash
git clone https://github.com/neuralet/neuralet.git
cd neuralet/applications/pose-estimation-tensorrt/
```

### Usage

##### Run on Jetson TX2
* You need to have [JetPack 4.3](https://developer.nvidia.com/jetpack-43-archive) installed on your Jetson TX2.

```bash
# 1) Download TensorRT engine file built with JetPack 4.3:
./download_engine.sh

# 2) Download/Copy Sample image
./download_sample_image.sh

# 3) Build Docker image for Jetson TX2 (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f jetson-tx2-openpifpaf.Dockerfile -t "neuralet/applications-openpifpaf:latest-jetson-tx2" .

# 4) Run Docker container:
docker run --runtime nvidia --privileged -it -v $PWD:/repo neuralet/applications-openpifpaf:latest-jetson-tx2
```

