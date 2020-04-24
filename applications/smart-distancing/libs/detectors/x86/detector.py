from libs.detectors.x86 import mobilenet_ssd


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
        self.net = mobilenet_ssd.Detector(self.config)

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output

