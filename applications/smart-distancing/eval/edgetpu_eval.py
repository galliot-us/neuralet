#!/usr/bin/python3
import argparse
import cv2 as cv
import os
import _init_paths
from edgetpu.detector import Detector
from exporter import export_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--minscore', required=True)
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--img_size', required=True)
    args = parser.parse_args()
    det_engine = Detector(args)

    class_name = ['face', 'mask-face']
    det_path = 'eval_files/detresults/'

    image_size = tuple(map(int, args.img_size.split(',')))
    image_path = args.img_path
    for filename in os.listdir(image_path):
        if not (filename.endswith('.jpg') or filename.endswith('jpeg')): continue
        img_path = os.path.join(image_path, filename)
        cv_image = cv.imread(img_path)
        resized_image = cv.resize(cv_image, tuple(image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        nn_out = det_engine.inference(rgb_resized_image)
        export_results(nn_out, class_name, det_path, filename, cv_image)
