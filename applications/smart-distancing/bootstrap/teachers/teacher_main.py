import sys
import time

import model_builder

sys.path.append("../../")
from libs.config_engine import ConfigEngine
import argparse
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = ConfigEngine(args.config)
    teacher_model = model_builder.build(config)
    video_uri = config.get_section_dict("Teacher")["VideoPath"]
    input_cap = cv.VideoCapture(video_uri)

    if (input_cap.isOpened()):
        print('opened video ', video_uri)
    else:
        print('failed to load video ', video_uri)
        exit(0)
    frame_num = 0
    while input_cap.isOpened():
        t_begin = time.perf_counter()
        _, cv_image = input_cap.read()
        preprocessed_image = teacher_model.preprocessing(cv_image)
        raw_results = teacher_model.inference(preprocessed_image)
        postprocessed_results = teacher_model.postprocessing(raw_results)
        teacher_model.save_results(cv_image, postprocessed_results)
        t_end = time.perf_counter()
        print("processed frame number {} in {} seconds".format(str(frame_num), str(round(t_end - t_begin, 2))),
              end="\r")
        frame_num += 1
    input_cap.release()
