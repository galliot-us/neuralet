# docker can be installed on the dev board following these instructions: 
# https://docs.docker.com/install/linux/docker-ce/debian/#install-using-the-repository , step 4: arm64
# 1) build: docker build -f Dockerfile-coral-dev-board -t "neuralet/coral-dev-board:applications-smart-distancing" .
# 2) run: docker run -it --privileged -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/coral-dev-board:applications-smart-distancing

FROM arm64v8/debian:buster

VOLUME  /repo

RUN apt-get update && apt-get install -y wget gnupg \
    && rm /etc/apt/sources.list  && rm -rf /var/lib/apt/lists \
    && wget -qO - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

COPY data/multistrap* /etc/apt/sources.list.d/

RUN apt-get update && apt-get install -y python3-pip pkg-config libedgetpu1-std 
# Also if you needed tensorflow: python-dev python3-dev libhdf5-dev python3-h5py python3-scipy 
# Also python3-opencv may be needed, but it brings lots of dependencies (even x11-common !)

RUN python3 -m pip install --upgrade pip==19.3.1 setuptools==41.0.0 && python3 -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_aarch64.whl  
#if you needed tensorflow: grpcio==1.26.0  keras==2.2.4 protobuf https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.0.0/tensorflow-2.0.0-cp37-none-linux_aarch64.whl

RUN apt-get install -y python3-wget

RUN apt-get install -y python3-flask python3-opencv python3-scipy

WORKDIR /repo/applications/smart-distancing
# Also if you use opencv: LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0"

ENTRYPOINT ["python3", "neuralet-distancing.py"]
CMD ["--config", "config-skeleton.ini"]
