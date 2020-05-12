import time

import numpy as np

import cv2 as cv

from libs.detectors.utils.fps_calculator import convert_infr_time_to_fps

from openvino.inference_engine import IENetwork, IECore

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
        # Frames Per Second
        self.fps = None

        model_path = '/opt/intel/openvino_2020.2.120/deployment_tools/open_model_zoo/tools/downloader/intel/pedestrian-detection-adas-0002/FP32'

        core = IECore()
        network = core.read_network(
            model='{}/person-detection-retail-0013.xml'.format(model_path),
            weights='{}/person-detection-retail-0013.bin'.format(model_path)
        )
        self.input_layer = next(iter(network.inputs))
        self.detection_model = core.load_network(network=network, device_name='CPU')

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
        output = self.detection_model.infer(
            inputs={self.input_layer: input_image}
        )['detection_out']
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)

        class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])
        result = []

        for i, (_, label, score, x_min, y_min, x_max, y_max) in enumerate(output[0][0]):
            box = [y_min, x_min, y_max, x_max]
            if label == class_id and score > score_threshold:
                result.append({"id": str(class_id) + '-' + str(i), "bbox": box, "score": score})

        return result

