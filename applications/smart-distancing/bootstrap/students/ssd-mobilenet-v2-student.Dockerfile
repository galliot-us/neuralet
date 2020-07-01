#This container will install TensorFlow Object Detection API and its dependencies in the /model/research/object_detection directory

FROM tensorflow/tensorflow:1.15.0-gpu-py3
VOLUME /repo
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y git protobuf-compiler python3-tk vim wget

RUN pip install Cython && \
    pip install contextlib2 && \
    pip install pillow && \
    pip install lxml && \
    pip install jupyter && \
    pip install matplotlib

RUN git clone https://github.com/tensorflow/models.git && cd models && \
    git checkout 02c7112eb7ff0aed28d8d508708b3fb3a9c9c01f && cd ../

RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    make && \
    cp -r pycocotools/ /models/research/

RUN cd /models/research && \
    protoc object_detection/protos/*.proto --python_out=.

RUN echo 'export PYTHONPATH=$PYTHONPATH:/models/research:/models/research/slim' >> ~/.bashrc && \
    echo 'export export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc && \
    source ~/.bashrc

WORKDIR /repo/applications/smart-distancing/bootstrap/students
ENTRYPOINT ["bash", "./train_student.sh"]
CMD ["-c", "/repo/applications/smart-distancing/bootstrap/configs/iterdet.ini"]
