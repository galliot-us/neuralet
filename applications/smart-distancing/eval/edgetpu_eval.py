#!/usr/bin/python3
import argparse
import cv2 as cv
import os
import _init_paths
from edgetpu.detector import Detector
from exporter import export_results

def read_class_name(file_path):
    with open(file_path) as f:
        class_name=[word.split(':')[1] for line in f for word in line.split()]
    return class_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--minscore', required=True)
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--img_size', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('-gt', '--groundtruths', required=True)
    parser.add_argument('-t', '--threshold', required=True)
    args = parser.parse_args()
    det_engine = Detector(args)

    class_name = read_class_name(args.classes)
    det_path = args.result_dir
    if not os.path.exists(det_path):
        os.mkdir(det_path)
 
    image_size = tuple(map(int, args.img_size.split(',')))
    image_path = args.img_path
    print('Start exporting results at "{}"'.format(det_path))
    for filename in os.listdir(image_path):
        if not (filename.endswith('.jpg') or filename.endswith('jpeg')): continue
        img_path = os.path.join(image_path, filename)
        cv_image = cv.imread(img_path)
        resized_image = cv.resize(cv_image, tuple(image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        nn_out = det_engine.inference(rgb_resized_image)
        export_results(nn_out, class_name, det_path, filename.split('.')[0], cv_image)
    print('Detection results are created!')
    print('=============================================')
    print('Start evaluating the model...')
    os.system('python3 pascal_evaluator.py -gt {} -det {} -t {}'.format(args.groundtruths, det_path, args.threshold))
