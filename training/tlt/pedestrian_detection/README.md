# Train a Pedestrian Detection Model with Oxford Town Center Dataset in Nvidia Transfer Learning Toolkit

This is a set of scripts, config files and a Jupyter notebook that guide you to train a pedestrian detection with [Nvidia Transfer Learning Toolkit](https://developer.nvidia.com/transfer-learning-toolkit) (TLT).
For a comprehensive guide of TLT visit [here](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/index.html)
Follow these steps to train the model:
1. Create an account in https://ngc.nvidia.com/ and get an API key
2. Install [Docker](https://docs.docker.com/engine/install/debian/) in your local machine.
3. Install [Nvidia Docker 2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) in your local machine.
4. Install Nvidia GPU driver v410.xx or above
5. Run followings to clone the Neuralet repository: 
``` 
git clone https://github.com/neuralet/neuralet.git 
cd neuralet
```
6. Create a directory that will save all of the data and model's weights:
```
mkdir pedestrian_detection_tlt
```
7. Pull and run the TLT docker container by running:
```
docker run --runtime=nvidia -it -v [EXPERIMENT DIRECTORY]:/experiment_dir \
-v [PATH TO CLONED REPOSITORY]:/repo -p 8888:8888 nvcr.io/nvidia/tlt-streamanalytics:v2.0_dp_py2>
```
8. Install the dependencies:
```
apt-get update && apt-get install -y pkg-config libsm6 libxext6 libxrender-dev ffmpeg
pip3 install opencv-python pandas==0.24.0
```
10. Move to the repository TLT directory
```
cd /repo/training/tlt/pedestrian_detection
```
10. Execute Jupyter Notebook:
```
jupyter notebook --ip 0.0.0.0 --allow-root
```
11. Copy and paste the the link into your browser to open the Jupyter Notebook.
12. open the `pedestrian_ssd_mobilenet_v2.ipynb` and follow it to train the model.

