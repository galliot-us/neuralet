import sys
import time
import os
import model_builder

sys.path.append("../../")
from libs.config_engine import ConfigEngine
import argparse
import cv2 as cv
import logging

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = ConfigEngine(args.config)
    teacher_model = model_builder.build(config)
    video_uri = config.get_section_dict("Teacher")["VideoPath"]
    image_dir = config.get_section_dict("Teacher")["ImagePath"]
    max_allowed_image = int(config.get_section_dict("Teacher")["MaxAllowedImage"])
    input_cap = cv.VideoCapture(video_uri)
    if (input_cap.isOpened()):
        print('opened video ', video_uri)
    else:
        print('failed to load video ', video_uri)
        exit(0)
    frame_num = 0
    total_infer_time = 0
    logging.info("Teacher Inference Process Started")
    while input_cap.isOpened():
        number_of_existing_img = len(
            [name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        if number_of_existing_img > max_allowed_image:
            logging.info("Maximum allowed image to store exceeds, Teacher will stop inference for 1 hour")
            time.sleep(3600)
        t_begin = time.perf_counter()
        ret, cv_image = input_cap.read()
        if not ret:
            logging.info('Processed all frames')
            break
        preprocessed_image = teacher_model.preprocessing(cv_image)
        raw_results = teacher_model.inference(preprocessed_image)
        postprocessed_results = teacher_model.postprocessing(raw_results)
        teacher_model.save_results(cv_image, postprocessed_results)
        t_end = time.perf_counter()
        total_infer_time += round(t_end - t_begin, 2)
        if frame_num % 100 == 0:
            logging.info("processed frame number {} in {} seconds".format(str(frame_num), str(round(total_infer_time, 2))))
            total_infer_time = 0
        frame_num += 1
    input_cap.release()
