import time
import os

import numpy as np

import tensorflow as tf

from smart_distancing.utils.fps_calculator import convert_infr_time_to_fps

import smart_distancing as sd

__all__ = [
    'TfDetector',
]

class TfDetector(sd.detectors.BaseDetector):
    """
    TfDetector is a TensorFlow implementation of BaseDetector. It will use
    tf.keras.utils.get_file to get a model of the name chosen in the config file.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    SUPPORTED_PLATFORMS = sd.detectors.x86.PLATFORM_STRING
    DEFAULT_MODEL_URL = "http://download.tensorflow.org/models/object_detection/"

    # these properties are not used on this detector
    model_path = None
    model_file = None
    model = None
    # we use an attribute, since a getter/setter is not necessary
    fps = 0

    def __init__(self, config):
        """
        Implementation of __init__ that skips the download step because
        this step is performed by tf.keras.utils.get_file in load_model()
        """
        # set the config
        self.config = config

        # load the model
        self.load_model()

    def load_model(self):
        keras_dl_root = tf.keras.utils.get_file(
            fname=self.name,
            origin=self.DEFAULT_MODEL_URL + self.DEFAULT_MODEL_FILE,
            untar=True)

        self.model_path = os.path.join(keras_dl_root, "saved_model")

        self.model = tf.saved_model.load(self.model_path)
        self.model = self.model.signatures['serving_default']


    def inference(self, resized_rgb_image):
        """
        inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding detection output which is used for creating result
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)

        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        input_image = np.expand_dims(resized_rgb_image, axis=0)
        input_tensor = tf.convert_to_tensor(input_image)
        t_begin = time.perf_counter()
        output_dict = self.model(input_tensor)
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)

        boxes = output_dict['detection_boxes']
        labels = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])
        result = []
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == class_id and scores[0, i] > score_threshold:
                result.append({"id": str(class_id) + '-' + str(i), "bbox": boxes[0, i, :], "score": scores[0, i]})

        self.on_frame(result)
