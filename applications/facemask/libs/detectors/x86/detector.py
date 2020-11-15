class Detector:
    """
    Detector class is a high level class for detecting object using x86 devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """
    def __init__(self, config):
        self.config = config
        self.name = self.config.DETECTOR_NAME

        if self.name == "openpifpaf":
            from libs.detectors.x86 import openpifpaf
            self.net = openpifpaf.Detector(self.config)
        elif self.name == "tinyface":
            from libs.detectors.x86 import tinyface
            self.net = tinyface.Detector(self.config)
        else:
            raise ValueError('Not supported network named: ', self.name)

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output
