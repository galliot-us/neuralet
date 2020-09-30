from configs.config_handler import Config
from libs.classifiers.x86.classifier import Classifier
from libs.detectors.x86.detector import Detector
from models.frontend import FacemaskClassifierModel


def main():
    config_path = 'configs/config.json'
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)
    classifier = FacemaskClassifierModel(cfg)
    model = classifier.model

    cls_model = Classifier(model, cfg)
    import numpy as np

    # resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
    # rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    rgb_resized_image = np.random.random(size=[cfg.DETECTOR_INPUT_SIZE, cfg.DETECTOR_INPUT_SIZE, 3])
    object_list = Detector.inference(rgb_resized_image)
    print(object_list)

    img = np.random.random(size=[2, 45, 45, 3])
    result = cls_model.inference(img)
    print(result)


if __name__ == "__main__":
    main()
