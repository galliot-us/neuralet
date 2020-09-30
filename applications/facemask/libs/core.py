import cv2 as cv
import numpy as np
from libs.detectors.x86.detector import Detector


class FaceMaskAppEngine:

    def __init__(self, config):
        self.config = config
        self.ui = None
        self.detector = None
        self.running_video = False

        self.detector = Detector(self.config)
        self.image_size = (self.config.DETECTOR_INPUT_SIZE, self.config.DETECTOR_INPUT_SIZE, 3)

    def set_ui(self, ui):
        self.ui = ui

    def __process(self, cv_image):
        # Resize input image to resolution
        resolution = self.config.APP_VIDEO_RESOLUTION
        cv_image = cv.resize(cv_image, tuple(resolution))

        resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        objects_list = self.detector.inference(rgb_resized_image)
        [w, h] = resolution

        for obj in objects_list:
            box = obj["bbox"]
            x0 = box[1]
            y0 = box[0]
            x1 = box[3]
            y1 = box[2]
            obj["bbox"] = [x0, y0, x1, y1]
            obj["bboxReal"] = [x0 * w, y0 * h, x1 * w, y1 * h]

        return cv_image, objects_list

    def process_video(self, video_uri):
        input_cap = cv.VideoCapture(video_uri)

        if (input_cap.isOpened()):
            print('opened video ', video_uri)
        else:
            print('failed to load video ', video_uri)
            return

        self.running_video = True
        while input_cap.isOpened() and self.running_video:
            _, cv_image = input_cap.read()
            if np.shape(cv_image) != ():
                cv_image, objects = self.__process(cv_image)
            else:
                continue
            self.ui.update(cv_image, objects)
        input_cap.release()
        self.running_video = False

    # def process_image(self, image_path):
    #     # Process and pass the image to ui modules
    #     cv_image = cv.imread(image_path)
    #     cv_image, objects, distancings = self.__process(cv_image)
    #     self.ui.update(cv_image, objects, distancings)
