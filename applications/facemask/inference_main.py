from configs.config_handler import Config
from libs.classifiers.x86.classifier import Classifier
from libs.detectors.x86.detector import Detector
from models.frontend import FacemaskClassifierModel
from libs.core import FaceMaskAppEngine as CvEngine
from ui.web_gui import WebGUI as UI


def main():
    config_path = 'configs/config.json'
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)
    classifier = FacemaskClassifierModel(cfg)
    model = classifier.model
    
    engine = CvEngine(model, cfg)
    ui = UI(cfg, engine)
    engine.set_ui(ui)
    ui.start()


if __name__ == "__main__":
    main()
