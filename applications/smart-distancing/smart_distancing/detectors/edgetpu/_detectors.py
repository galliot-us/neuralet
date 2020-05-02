import time
import logging

from typing import List

import numpy as np

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

from smart_distancing.utils.fps_calculator import convert_infr_time_to_fps

import smart_distancing as sd

logger = logging.getLogger(__name__)

TFLITE_CLASS_DOC = """
Perform object detection with the given model. The model is a quantized tflite
file which if the detector can not find it at the path it will download it
from neuralet repository automatically.

:param config: Is a ConfigEngine instance which provides necessary parameters.
"""

class EdgeTpuDetector(sd.detectors.BaseDetector):
    """
    A base class for edgetpu (Coral) Detectors. The following should be
    overridden:

    DEFAULT_MODEL_FILE with the desired model filename
    """

    PLATFORM = 'edgetpu'
    DEFAULT_MODEL_URL = 'https://github.com/google-coral/edgetpu/raw/master/test_data/'

    # set on load_model()
    interpreter = None  # type: Interpreter
    # to store the fps for the fps property
    _fps = None  # type: int
    _sources = None  # type: List[str]

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def input_details(self):
        return self.interpreter.get_input_details()

    @property
    def output_details(self):
        return self.interpreter.get_output_details()

    @property
    def sources(self) -> List[str]:
        # TODO(mdegans): implemment this, and the setter
        logger.warning("getting sources not implemented")
        return self._sources

    @sources.setter
    def sources(self, sources: List[str]):
        logger.warning("setting sources not implemented")
        self._sources = sources

    def load_model(self):
        # Load TFLite model and allocate tensors
        self.interpreter = Interpreter(self.model_file, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.interpreter.allocate_tensors()

    def inference(self, resized_rgb_image: np.ndarray):
        """
        Sets input tensor to input resized_rgb_image and gets the output.
        The interpreter instance provides corresponding detection output which is used for creating result

        Args:
            resized_rgb_image (:obj:`np.ndarray`): uint8 ndarray with shape (img_height, img_width, channels) (HWC)

        Returns:
            result(:obj:`dict`): a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        input_image = np.expand_dims(resized_rgb_image, axis=0)
        # Fill input tensor with input_image
        self.interpreter.set_tensor(self.input_details[0]["index"], input_image)
        t_begin = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - t_begin  # Second
        self._fps = convert_infr_time_to_fps(inference_time)
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        labels = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        # TODO: will be used for getting number of objects
        # num = self.interpreter.get_tensor(self.output_details[3]['index'])

        result = []
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == self.class_id and scores[0, i] > self.score_threshold:
                result.append({"id": str(self.class_id) + '-' + str(i), "bbox": boxes[0, i, :], "score": scores[0, i]})

        # call on_frame with the detections
        self.on_frame(result)


class MobilenetSsdDetector(EdgeTpuDetector):
    __doc__ = TFLITE_CLASS_DOC
    DEFAULT_MODEL_FILE = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'


class PedestrianSsdDetector(EdgeTpuDetector):
    __doc__ = TFLITE_CLASS_DOC
    DEFAULT_MODEL_FILE = 'ped_ssd_mobilenet_v2_quantized_edgetpu.tflite'


class PedestrianSsdLiteDetector(EdgeTpuDetector):
    __doc__ = TFLITE_CLASS_DOC
    DEFAULT_MODEL_FILE = 'ped_ssdlite_mobilenet_v2_quantized_edgetpu.tflite'
