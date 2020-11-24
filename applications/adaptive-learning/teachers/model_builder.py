from iterdet import IterDet
from faster_rcnn_nas import FasterRcnnNas
import logging
TEACHER_MAP = {
    "iterdet": IterDet,
    "faster_rcnn_nas": FasterRcnnNas
               }


def build(config):
    """ build a teacher model based on config file"""
    model_name = config.get_section_dict('Teacher')['Name']
    if model_name in TEACHER_MAP.keys():
        model = TEACHER_MAP[model_name]
        logging.info("model loaded successfully")
    else:
        raise ValueError("The Teacher model name should be iterdet but {} provided".format(model_name))
    return model(config=config)
