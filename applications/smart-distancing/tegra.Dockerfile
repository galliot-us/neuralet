# docker is pre-installed on Tegra devices
# 1) build: (sudo) docker build -f tegra.Dockerfile -t "neuralet/jetson-nano:applications-smart-distancing" .
# 2) run: (sudo) docker run -it --runtime nvidia -p HOST_PORT:8000 neuralet/jetson-nano:applications-smart-distancing

# this is l4t-base with the apt sources enabled
# the lack of apt sources seems to be an oversight on the part of Nvidia
# it should be unnecessary to do this in later releases.
FROM registry.hub.docker.com/mdegans/l4t-base:r32.3.1

ARG DEBIAN_FRONTEND=noninteractive

# install runtime depdenencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-flask \
        python3-opencv \
        python3-scipy \
        python3-matplotlib \
        python3-pillow \
    && rm -rf /var/lib/apt/lists/*

# copy just libflattenconcat build script initially
COPY smart_distancing/data/scripts/build_libflattenconcat.sh /tmp/

# install build deps for stuff that needs building, remove build deps
# libnvinfer-samples provides /usr/src/tensorrt for libflattenconcat.so
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        cuda-minimal-build-10-0 \
        libnvinfer-samples \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-libnvinfer \
    && pip3 install pycuda \
    && useradd -mrd /var/smart_distancing smart_distancing \
    && mkdir -p /var/smart_distancing/.smart_distancing/plugins/ \
    && chown -R smart_distancing:smart_distancing /var/smart_distancing/.smart_distancing \
    && /tmp/build_libflattenconcat.sh /var/smart_distancing/.smart_distancing/plugins/ \
    && apt-get purge -y --autoremove \
        cmake \
        cuda-minimal-build-10-0 \
        libnvinfer-samples \
        python3-dev \
        python3-pip \
        python3-setuptools \
    && rm -rf /var/lib/apt/lists/*
# the apt packages are build dependencies, but not runtime dependencies
# since --runtime nvidia bind mounts libs at runtime, so we can purge
# these packages after we're done installing pycuda

# copy source last, so it can be modified easily without rebuilding everything.
COPY . /repo/
WORKDIR /repo/

# run as smart distancing and the video group, which can access the GPU
# (insted of running as root). The base image is also defanged.
USER smart_distancing:video

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "smart_distancing", "--verbose"]
CMD ["--config", "jetson.ini"]
