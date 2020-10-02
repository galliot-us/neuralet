from libs.detectors.x86.detecor import Detector
from libs.detectors.edgetpu.detector import Detector
from libs.classifiers.x86.classifier import Classifier
from libs.classifiers.edgetpu.classifier import Classifier
from configs.config_handler import Config
import cv2 as cv
import PIL
import numpy as np


def main():
    config_path = 'configs/config.json'
    input_path = ''
    output_path = ''

    cfg = Config(path=config_path)
    detector_input_size = (cfg.DETECTOR_INPUT_SIZE[0], cfg.DETECTOR_INPUT_SIZE[1], 3)

    device = cfg.DEVICE
    detector = None
    classifier = None

    if device == "x86":
        from libs.detectors.x86.detector import Detector
        from libs.classifiers.x86.classifier import Classifier
        detector = Detector(cfg)
        classifier_model = Classifier(cfg)
    elif device == "EdgeTPU":
        from libs.detectors.edgetpu.detector import Detector
        from libs.classifiers.edgetpu.classifier import Classifier
        detector = Detector(cfg)
        classifier_model = Classifier(cfg)
    else:
        raise ValueError('Not supported device named: ', device)

    image_size = (cfg.DETECTOR_INPUT_SIZE[0], cfg.DETECTOR_INPUT_SIZE[1], 3)
    classifier_img_size = (cfg.CLASSIFIER_INPUT_SIZE, cfg.CLASSIFIER_INPUT_SIZE, 3)

    input_cap = cv.VideoCapture(input_path)
    while (input_cap.isOpened()):
        ret, raw_img = input_cap.read()
        if ret == False:
            break
        _, cv_image = input_cap.read()
        if np.shape(cv_image) != ():
            resized_image = cv.resize(cv_image, tuple(detector_input_size[:2]))
            rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
            objects_list = detector.inference(rgb_resized_image)
            print(objects_list)
        else:
            continue


if __name__ == '__main__':
    main()
