
class Classifier:
    """
    Classifier class is a high-level class for classifying images using EdgeTPU devices.
    When an instance of the Classifier is created you can call inference method and feed your
    input image in order to get the classifier results.
    :param config: Is a Config instance which provides necessary parameters.
    """
    def __init__(self, config):
        self.config = config
        self.name = self.config.CLASSIFIER_NAME

        if self.name == 'OFMClassifier':
            from libs.classifiers.edgetpu import face_mask_edgetpu
            self.net = face_mask_edgetpu.Classifier(self.config)
        else:
            raise ValueError('Not supported network named: ', self.name)

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output, scores = self.net.inference(resized_rgb_image)
        return output, scores
