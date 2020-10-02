FROM amd64/debian:buster

VOLUME  /repo

RUN apt-get update && apt-get install -y wget gnupg usbutils


ENV PYTHONPATH=$PYTHONPATH:/project-posenet

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN wget -qO - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && apt-get install -y python3-pip pkg-config libedgetpu1-std python3-wget python3-edgetpu git

RUN git clone \
    https://github.com/google-coral/project-posenet.git && sed -i 's/sudo / /g' \
    /project-posenet/install_requirements.sh && sh /project-posenet/install_requirements.sh

ENV PYTHONPATH=$PYTHONPATH:/project-posenet

RUN python3 -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl 

RUN apt-get install -y python3-flask python3-opencv python3-scipy

WORKDIR /repo/applications/facemask
