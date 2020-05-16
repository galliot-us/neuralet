FROM node:14-alpine as frontend
WORKDIR /frontend
COPY ui/frontend/package.json ui/frontend/package-lock.json /frontend/
RUN npm install --production
COPY ui/frontend /frontend
RUN npm run build

FROM openvino/ubuntu18_runtime
ARG OPENCV_VERSION=4.3.0
USER root
VOLUME  /repo
WORKDIR /repo/applications/smart-distancing

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++ \
        gcc \
        gstreamer1.0-plugins-good \
        libavcodec-dev \
        libavformat-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libsm6 \
        libswscale-dev \
        libxext6 \
        libxrender-dev \
        pkg-config \
        python3-numpy \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && curl https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -o opencv.tar.gz -L \
    && tar zxvf opencv.tar.gz && rm opencv.tar.gz \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mkdir opencv/build \
    && cd opencv/build \
    && cmake -DBUILD_opencv_python3=yes -DPYTHON_EXECUTABLE=/usr/local/bin/python3 ../ \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && rm -rf opencv

RUN pip3 install --upgrade pip setuptools==41.0.0 && pip3 install wget fastapi uvicorn aiofiles scipy image

COPY --from=frontend /frontend/build /srv/frontend

EXPOSE 8000

CMD source /opt/intel/openvino/bin/setupvars.sh && python3 neuralet-distancing.py --config=config-x86-openvino.ini

