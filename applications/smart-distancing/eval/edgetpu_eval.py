#!/usr/bin/python3
import argparse
import cv2 as cv
import os
import _init_paths
from edgetpu.detector import Detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--minscore', required=True)
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--img_size', required=True)
    args = parser.parse_args()
    det_engine = Detector(args)

    image_size = tuple(map(int, args.img_size.split(',')))
    image_path = args.img_path
    for filename in os.listdir(image_path):
        if not (filename.endswith('.jpg') or filename.endswith('jpeg')): continue
        img_path = os.path.join(image_path, filename)
        cv_image = cv.imread(img_path)
        resized_image = cv.resize(cv_image, tuple(image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        tmp_objects_list = det_engine.inference(rgb_resized_image)
