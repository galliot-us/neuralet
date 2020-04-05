
class Detector():
    def __init__(self, config):
        self.config = config
        self.net = None 

        if self.config.get_section_dict('Detector')['Name'] == 'mobilenet_ssd_v2':
            self.net = None # this should be TrtSSD or whatever ... 

    def inference(self, cv_image):
        # Here should inference on the image, output a list of objects, each obj is a dict with two keys "id" and "bbox" 
        # return [{"id": 0, "bbox": [x, y, w, h]}, {...}, {...}, ...]
