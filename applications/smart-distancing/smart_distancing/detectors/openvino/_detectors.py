import os
import logging
import time
import urllib.parse
import urllib.request

import numpy as np
import cv2 as cv

import smart_distancing as sd

from openvino.inference_engine import IECore

from typing import (
    Dict,
)

__all__ = ['OpenVinoDetector']

logger = logging.getLogger(__name__)

class OpenVinoDetector(sd.detectors.BaseDetector):
    """
    Perform object detection with the given model. The model is a quantized tflite
    file which if the detector can not find it at the path it will download it
    from neuralet repository automatically.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    PLATFORM = 'openvino'

    DEFAULT_MODEL_URL = 'https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/person-detection-retail-0013/FP32/'
    DEFAULT_MODEL_XML = 'person-detection-retail-0013.xml'
    DEFAULT_MODEL_BIN = 'person-detection-retail-0013.bin'

    _input_layer = None
    _detection_model = None

    _fps = None  # type: int

    @property
    def model_files(self) -> Dict[str, str]:
        return {
            'model': os.path.join(self.model_path, self.DEFAULT_MODEL_XML),
            'weights': os.path.join(self.model_path, self.DEFAULT_MODEL_BIN),
        }

    @property
    def model_urls(self) -> Dict[str, str]:
        return {
            'model': urllib.parse.urlunparse(urllib.parse.urlparse(
                self.DEFAULT_MODEL_URL + self.model_files['model'])),
            'weights': urllib.parse.urlunparse(urllib.parse.urlparse(
                self.DEFAULT_MODEL_URL + self.model_files['weigths'])),
        }

    def __init__(self, config):
        # set the config
        self.config = config

        # download the model if necessary
        for name, filename in self.model_files.items():
            if not os.path.isfile(filename):
                logger.info(
                    f'model does not exist under: "{filename}" '
                    f'downloading from  "{self.model_urls[name]}"')
                os.makedirs(self.model_path, mode=0o755, exist_ok=True)
                urllib.request.urlretrieve(self.model_urls[name], filename)

        # load the model
        self.load_model()

    def load_model(self):
        core = IECore()
        network = core.read_network(
            model=self.model_files['model'],
            weights=self.model_files['weights'],
        )
        self._input_layer = next(iter(network.inputs))
        self._detection_model = core.load_network(
            network=network,
            device_name='CPU',
        )

    def inference(self, resized_rgb_image):
        """
        inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding detection output which is used for creating result
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)

        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """

        required_image_size = (544, 320)

        input_image = cv.resize(resized_rgb_image, required_image_size)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)

        t_begin = time.perf_counter()
        output = self._detection_model.infer(
            inputs={self._input_layer: input_image}
        )['detection_out']
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self._fps = sd.utils.fps_calculator.convert_infr_time_to_fps(
            inference_time)

        class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])
        result = []

        for i, (_, label, score, x_min, y_min, x_max, y_max) in enumerate(output[0][0]):
            box = [y_min, x_min, y_max, x_max]
            if label == class_id and score > score_threshold:
                result.append({"id": str(class_id) + '-' + str(i), "bbox": box, "score": score})

        # call on_frame with the detections
        self.on_frame(result)

    @property
    def fps(self):
        return self._fps
