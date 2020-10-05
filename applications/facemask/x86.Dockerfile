FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

VOLUME  /repo
WORKDIR /repo/applications/facemask

RUN apt-get update && apt-get install -y pkg-config libsm6 libxext6 libxrender-dev libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget flask scipy image tensorflow-gpu==1.15.0 openpifpaf keras==2.3.0 matplotlib scikit-learn
