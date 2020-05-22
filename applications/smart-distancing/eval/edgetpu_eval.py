#!/usr/bin/python3
import argparse

from eval.libs.edgetpu.detector import Detector
import cv2 as cv
import os
import os, sys
from libs.config_engine import ConfigEngine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    image_path = 'val_images'
    args = parser.parse_args()
    config = ConfigEngine(args.config)
    image_size = [int(i) for i in config.get_section_dict('Detector')['ImageSize'].split(',')]
    det_engine = Detector(config)
    for filename in os.listdir(image_path):
        if not (filename.endswith('.jpg') or filename.endswith('jpeg')): continue
        img_path = os.path.join(image_path, filename)
        cv_image = cv.imread(img_path)
        resized_image = cv.resize(cv_image, tuple(image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        tmp_objects_list = det_engine.inference(rgb_resized_image)
        print(tmp_objects_list)
