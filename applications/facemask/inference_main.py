from configs.config_handler import Config
from libs.classifiers.x86.classifier import Classifier

def main():
    config_path = 'configs/config.json'
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)
    cls_model = Classifier(cfg)
    import numpy as np

    img = np.random.random(size=[1,45, 45, 3])
    result = cls_model.inference(img)
    print(img)

