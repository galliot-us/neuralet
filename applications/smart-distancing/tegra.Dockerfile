# docker is pre-installed on Tegra devices
# 1) build: (sudo) docker build -f tegra.Dockerfile -t "neuralet/jetson-nano:applications-smart-distancing" .
# 2) run: (sudo) docker run -it --runtime nvidia -p HOST_PORT:8000 neuralet/jetson-nano:applications-smart-distancing

# this is l4t-base with the apt sources enabled
# the lack of apt sources seems to be an oversight on the part of Nvidia
# it should be unnecessary to do this in later releases.
FROM registry.hub.docker.com/mdegans/l4t-base:latest

ARG DEBIAN_FRONTEND=noninteractive

# install runtime depdenencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-flask \
        python3-opencv \
        python3-scipy \
        python3-matplotlib \
        python3-pillow \
    && rm -rf /var/lib/apt/lists/*

# install build deps for pycuda, install pycuda, remove build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-setuptools \
        cuda-minimal-build-10-0 \
    && pip3 install pycuda \
    && apt-get purge -y --autoremove \
        python3-pip \
        python3-dev \
        python3-setuptools \
        cuda-minimal-build-10-0 \
    && rm -rf /var/lib/apt/lists/*
# the apt packages are build dependencies, but not runtime dependencies
# since --runtime nvidia bind mounts libs at runtime, so we can purge
# these packages after we're done installing pycuda


# copy source last, so it can be modified easily without rebuilding everything.
COPY . /repo/
WORKDIR /repo/

# create a regular user to run the app instead of default (root)
# /home/dir is /var/smart_distancing
# video group is required to access the GPU without root
RUN useradd -G video -mrd /var/smart_distancing smart_distancing
USER smart_distancing:smart_distancing

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "smart_distancing", "--verbose"]
CMD ["--config", "jetson.ini"]
