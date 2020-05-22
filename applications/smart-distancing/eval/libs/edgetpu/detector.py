import os
import time
import numpy as np

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter


class Detector:
    """
    Perform object detection with the given model. The model is a quantized tflite
    file which if the detector can not find it at the path it will download it
    from neuralet repository automatically.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        # Get the model name from the config
        self.model_name = self.config.get_section_dict('Detector')['Name']
        self.model_path = 'libs/detectors/edgetpu/data/' + self.model_file

        # Load TFLite model and allocate tensors
        self.interpreter = Interpreter(self.model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.interpreter.allocate_tensors()
        # Get the model input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get class id from config
        self.class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        self.score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])

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
        # Fill input tensor with input_image
        self.interpreter.set_tensor(self.input_details[0]["index"], input_image)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        labels = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        result = []
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == self.class_id and scores[0, i] > self.score_threshold:
                result.append({"id": str(self.class_id) + '-' + str(i), "bbox": boxes[0, i, :], "score": scores[0, i]})

        return result