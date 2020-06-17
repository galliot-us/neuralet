from .iterdet import IterDet

TEACHER_MAP = {"iterdet": IterDet}


def build(config):
    model_name = config.get_section_dict('Detector')['Name']
    model = TEACHER_MAP[model_name]
    return model(config=config)
