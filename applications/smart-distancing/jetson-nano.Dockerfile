# docker can be installed on the dev board following these instructions:
# https://docs.docker.com/install/linux/docker-ce/debian/#install-using-the-repository , step 4: arm64
# 1) build: docker build -f Dockerfile-jetson-nano -t "neuralet/jetson-nano:applications-smart-distancing" .
# 2) run: docker run -it --runtime nvidia --privileged -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/jetson-nano:applications-smart-distancing

FROM nvcr.io/nvidia/l4t-base:r32.3.1

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

VOLUME /repo

RUN wget https://github.com/Tony607/jetson_nano_trt_tf_ssd/raw/master/packages/jetpack4.3/tensorrt.tar.gz -O /opt/tensorrt.tar.gz
RUN tar -xzf /opt/tensorrt.tar.gz -C /usr/local/lib/python3.6/dist-packages/

RUN wget https://github.com/Tony607/jetson_nano_trt_tf_ssd/raw/master/packages/jetpack4.3/libflattenconcat.so -O /opt/libflattenconcat.so

RUN apt-get update && apt-get install -y python3-pip pkg-config

RUN apt-get install -y python3-flask python3-opencv python3-scipy python3-matplotlib

RUN pip3 install pycuda

WORKDIR /repo/applications/smart-distancing

ENTRYPOINT ["python3", "neuralet-distancing.py"]
CMD ["--config", "config-jetson.ini"]
