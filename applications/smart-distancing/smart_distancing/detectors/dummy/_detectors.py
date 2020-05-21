import time

import numpy as np

import smart_distancing as sd


__all__ = ['DummyDetector']


class DummyDetector(sd.detectors.BaseDetector):
    """
    Detects Random bounding boxes

    Detector class is a high level class for detecting object using edgetpu devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def inference(self, resized_rgb_image):
        self.fps = np.random.choice([0.5, 1, 2])
        time.sleep(1.0 / self.fps)
        bbox_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]) * 0.5
        class_id = self.class_id
        self.on_frame([{
            'id': str(class_id)+"-"+str(i),
            'bbox': (bbox_transform @ np.random.rand(4)).tolist(),
            'cls': class_id
        } for i in range(np.random.randint(5))])
