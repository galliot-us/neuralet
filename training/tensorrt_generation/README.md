[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# TensorRT Engine Generation from a custom SSD-MobileNet-V2 


## Getting Started

You can read the [Deploying a Custom SSD-MobileNet-V2 Model on the NVIDIA Jetson Nano]() article on our website. The following instructions will help you get started.

### Prerequisites

**Hardware**
  * NVIDIA Jetson Nano

  **Software**
  * You should have [Docker](https://docs.docker.com/get-docker/) on your device.

  ### Install

  Make sure you have the prerequisites and then clone this repository to your local system by running this command:

  ```
  git clone https://github.com/neuralet/neuralet.git
  cd neuralet/applications/smart-distancing/
  ```

  ### Usage

  Make sure you have `Docker` installed on your device by following [these instructions](https://docs.docker.com/install/linux/docker-ce/debian).

  **Download Required Files**
  ```
  cd neuralet/training/tensorrt_generation/

  # Download sample retrained pedestrian detector model 
  ./download_model.sh
  ```

  **Run on Jetson Nano**
  * You need to have JetPack 4.3 installed on your Jetson Nano.

  ```
  cd neuralet/training/tensorrt_generation/


  # 1) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
  docker build -f Dockerfile-nkh -t "neuralet/l4t-tensorrt-conversion" .

  # 3) Run Docker container:
  docker run -it --runtime nvidia --privileged --network host -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/l4t-tensorrt-conversion:latest
  ```

  ### Configurations
  You can read and modify the configurations in `config.ini` file. Under the `[MODEL]` section, you can modify the model specs. You can specify model name and path, number of classes of model, objects min size, max size and input image dimension there. 
  There is an `InputOrder` parameter which determines order of loc_data, conf_data and priorbox_data of model, which is set equal to the “NMS” node input order in the .pbtxt file.

