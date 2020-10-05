class Detector:
    """
    Detector class is a high level class for detecting object using EdgeTPU devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """
    def __init__(self, config):
        self.config = config
        self.name = self.config.DETECTOR_NAME

        if self.name == "open_pifpaf_jetson":
            from libs.detectors.jetson import open_pifpaf_jetson
            self.net = open_pifpaf_jetson.Detector(self.config)
        else:
            raise ValueError('Not supported network named:{} on Jetson device '.format(self.name))

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output
