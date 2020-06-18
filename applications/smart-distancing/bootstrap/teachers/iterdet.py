import copy
import os
import time

import wget
from mmdet.apis import init_detector, inference_detector

from teacher_meta_arch import TeacherMetaArch


def load_model():
    base_path = "data/iterdet"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    config_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.py")
    if not os.path.isfile(config_file):
        url = "https://raw.githubusercontent.com/saic-vul/iterdet/master/" \
              "configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py"
        print('config file does not exist under: ', config_file, 'downloading from ', url)
        wget.download(url, config_file)

    checkpoint_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.pth")
    if not os.path.isfile(checkpoint_file):
        url = "https://github.com/saic-vul/iterdet/releases/download/v2.0.0/crowd_human_full_faster_rcnn_r50_fpn_2x.pth"
        print('checkpoint file does not exist under: ', checkpoint_file, 'downloading from ', url)
        wget.download(url, checkpoint_file)

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    return model


class IterDet(TeacherMetaArch):

    def __init__(self, config):
        super(IterDet, self).__init__(config=config)
        self.detection_model = load_model()

    def inference(self, preprocessed_image):
        output_dict = inference_detector(self.detection_model, preprocessed_image)
        class_id = int(self.config.get_section_dict('Teacher')['ClassID'])
        score_threshold = float(self.config.get_section_dict('Teacher')['MinScore'])
        result = []
        for i, box in enumerate(output_dict[0]):  # number of boxes
            if box[-1] > score_threshold:
                result.append({"id": str(class_id) + '-' + str(i),
                               "bbox": [box[0] / self.image_size[0], box[1] / self.image_size[1],
                                        box[2] / self.image_size[0], box[3] / self.image_size[1]],
                               "score": box[-1]})

        return result

    def postprocessing(self, raw_results):
        post_processed_results = copy.copy(raw_results)
        for bbox in raw_results:
            if (bbox["bbox"][2] - bbox["bbox"][0]) * (bbox["bbox"][3] - bbox["bbox"][1]) > 0.2:
                post_processed_results.remove(bbox)
        return post_processed_results
