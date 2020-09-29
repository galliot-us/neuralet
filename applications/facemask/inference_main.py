from configs.config_handler import Config
from libs.classifiers.x86.classifier import Classifier
from models.frontend import FacemaskClassifierModel


def main():
    config_path = 'configs/config.json'
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)
    classifier = FacemaskClassifierModel(cfg)
    model = classifier.model
    
    cls_model = Classifier(model, cfg)
    import numpy as np

    img = np.random.random(size=[2,45, 45, 3])
    result = cls_model.inference(img)
    print(result)

if __name__ == "__main__":
    main()
