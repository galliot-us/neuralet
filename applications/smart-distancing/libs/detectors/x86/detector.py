

class Detector:
    """
    Detector class is a high level class for detecting object using edgetpu devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.name = self.config.get_section_dict('Detector')['Name']

        if self.name == 'mobilenet_ssd_v2':
            from libs.detectors.x86 import mobilenet_ssd
            self.net = mobilenet_ssd.Detector(self.config)
        elif self.name == "openvino":
            from libs.detectors.x86 import openvino
            self.net = openvino.Detector(self.config)
        elif self.name == "pedestrian_ssd_mobilenet_v2":
            from libs.detectors.x86 import pedestrian_ssd_mobilenet_v2
            self.net = pedestrian_ssd_mobilenet_v2.Detector(self.config)
        else:
            raise ValueError('Not supported network named: ', self.name)

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output

