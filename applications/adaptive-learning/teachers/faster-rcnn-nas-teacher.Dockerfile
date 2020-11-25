FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

VOLUME  /repo

WORKDIR /repo/applications/adaptive-learning/teachers

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget scipy image lxml

ENTRYPOINT ["python", "teacher_main.py"]

CMD ["--config", "/repo/applications/adaptive-learning/configs/faster_rcnn_nas.ini"]
