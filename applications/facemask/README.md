# Face Mask Detection Application
Face-Mask is an open-source application that can detect and classify faces them to Face and Mask classes. The application is capable to run at x86, Coral Dev Board TPU, and amd64 node with attached usb edge TPU.


## Getting Started
The following instructions will help you get started.

### Prerequisites

#### Hardware
A host edge device. We currently support the following:

* Coral Dev Board
* AMD64 node with attached Coral USB Accelerator
* X86 node

#### Software

* You should have [Docker](https://docs.docker.com/get-docker/) on your device.

### Install
Make sure you have the prerequisites and then clone this repository to your local system by running this command:
```
git clone https://github.com/neuralet/neuralet
cd neuralet/applications/facemask/
```
## Usage
The application has a module for trainig Face-Mask classifier and three inference mode:
1- Inference on app mode
2- Inference on video
3- Inference on image
which are compatible with x86 devices, Coral USB Accelerator, and Coral Dev Borad.

* NOTE: There is a config file at `configs/` directory for customizing the parameters of the model and the application. Please set the parameters if you plan to have a customized setting  

### Run on x86
```
# 1) Build Docker image
docker build -f x86.Dockerfile -t "neuralet/face-mask:latest-x86" .

# 2) Run Docker container:
docker run  -it --gpus all -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-x86

# 3) Run main application python script
python inference_main_app.py --config configs/config.json 
```
### Run on AMD64 node with a connected Coral USB Accelerator
```
# 1) Build Docker image
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/face-mask:latest-amd64" .

# 2) Run Docker container:
docker run  -it --privileged -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-amd64

# 3) Run main application python script
python inference_main_app.py --config configs/config_edgetpu.json 
```
### Run on Coral Dev Board
```
# 1) Build Docker image
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/face-mask:latest-coral-dev-board" .

# 2) Run Docker container:
docker run  -it --privileged -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-coral-dev-board

# 3) Run main application python script
python inference_main_app.py --config configs/config_edgetpu.json 
```

### Train The Model
On X86 devices it is possible to train the face-mask classifier with new dataset, first step for training the model is creating a train and validation set and set like the below structure.
Ex:
```
# train/
# |_face
# |    |__face1.jpg
# |    |__face2.jpg
# |_face-mask
#      |__face-mask1.jpg
#      |__face-mask2.jpg

# validation/
# |_face
# |    |__face1.jpg
# |    |__face2.jpg
# |_face-mask
#      |__face-mask1.jpg
#      |__face-mask2.jpg
```
Set dataset path at `configs/config.json` file according and run the below script.

```
python model_main.py --config configs/config.json
```
### Inferencing On Video and Images
There are two easy-to-use scripts for inferencing on video and image.
- The following script get a video file as its input and export the output video at given path. 
`python inference_video.py --config configs/config.json --input_video_path data/video/sample.mov --output_video data/videos/output.avi`

- For inferencing on images run the below scrip.
`python inference_images.py --config configs/config.json --input_image_dir data/images --output_image_dir output_images`
