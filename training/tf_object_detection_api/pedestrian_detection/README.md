# Train Pedestrian Detection Model with TensorFlow Object Detection API

This is a set of scripts, config files and a Jupyter notebook that guide you to train a pedestrian detection with [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
Follow these steps to train the model:
1. Install [Docker](https://docs.docker.com/engine/install/debian/) in your local machine.
2. Install [Nvidia Docker 2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) in your local machine.
3. Run followings to clone the Neuralet repository: 
``` 
git clone https://github.com/neuralet/neuralet.git 
cd neuralet
```
4. Create a directory that will save all of the data and model's checkpoints:
```
mkdir pedestrian_detection
```
5. Pull and run the Neuralet TensorFlow Object Detection API docker container:
```
docker run --gpus all -it -v [EXPERIMENT DIRECTORY]:/experiment_dir \
-v [PATH TO CLONED REPOSITORY]:/repo -p 8888:8888 -p 6006:6006 neuralet/tools-tf-object-detection-api-training
```
6. Install the dependencies:
```
apt-get update && apt-get install -y wget ffmpeg
pip install pandas
```
7. Move to the repo's pedestrian detection directory
```
cd /repo/training/tf_object_detection_api/pedestrian_detection
```
8. Execute Jupyter Notebook:
```
jupyter notebook --ip 0.0.0.0 --allow-root
```
9. Copy and paste the the link into your browser to open the Jupyter Notebook.
10. open the `pedestrian_detection.ipynb` and follow it to train the model.

P.S. You can track the training procedure by executing the Neuralet TensorFlow Object Detection API docker container you ran earlier by `docker exec -it [container name] bash` and run:
```
tensorboard --logdir [PATH TO MODEL DIRECTORY]
```
And browse `localhost:6006` in your browser.

