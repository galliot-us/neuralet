#!/usr/bin/python3
import argparse
import cv2 as cv
import os
import _init_paths
from edgetpu.detector import Detector
from exporter import export_results


def read_class_name(file_path):
    """
    Read a .txt file and extract the classes

    Args:
        file_path: The path of txt file
    Returns:
        class_name: List of class names
    """
    with open(file_path) as f:
        class_name = [word.split(":")[1] for line in f for word in line.split()]
    return class_name


if __name__ == "__main__":
    # python edgetpu_eval.py --model_path PATH/model.tflite --classes PATH/cls.txt --minscore 0.25 --img_path TEST_IMG/ --img_size 300,300,3 --result_dir results/ -gt groundtruths -t 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--minscore", required=True)
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--img_size", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("-gt", "--groundtruths", required=True)
    parser.add_argument("-t", "--threshold", required=True)
    args = parser.parse_args()
    det_engine = Detector(args)

    # Get the class names
    class_name = read_class_name(args.classes)
    # The path of exporting detector results
    det_path = args.result_dir
    if not os.path.exists(det_path):
        os.mkdir(det_path)

    # The size of images based on the models input size
    image_size = tuple(map(int, args.img_size.split(",")))
    image_path = args.img_path
    print('Start exporting results at "{}"'.format(det_path))
    # Inference all images in image_path directory
    for filename in os.listdir(image_path):
        if not (filename.endswith(".jpg") or filename.endswith("jpeg")):
            continue
        img_path = os.path.join(image_path, filename)
        cv_image = cv.imread(img_path)
        resized_image = cv.resize(cv_image, tuple(image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        nn_out = det_engine.inference(rgb_resized_image)
        # Export result for each image at a txt file.
        export_results(nn_out, class_name, det_path, filename.split(".")[0], cv_image)
    print("Detection results are created!")
    print("=============================================")
    print("Start evaluating the model...")
    os.system(
        "python3 pascal_evaluator.py -gt {} -det {} -t {}".format(
            args.groundtruths, det_path, args.threshold
        )
    )
