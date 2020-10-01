from configs.config_handler import Config
from libs.core import FaceMaskAppEngine as CvEngine
from ui.web_gui import WebGUI as UI
from argparse import ArgumentParser


def main():
    """
    Creates config and application engine module and starts ui
    :return:
    """
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='json config file path', default='configs/config.json')
    args = argparse.parse_args()
    config_path = args.config
    print("-_- -_- -_- -_- -_- -_- -_- Running %s -_- -_- -_- -_- -_- -_- -_-" % config_path)
    cfg = Config(path=config_path)

    engine = CvEngine(cfg)
    ui = UI(cfg, engine)
    engine.set_ui(ui)
    ui.start()


if __name__ == "__main__":
    main()
