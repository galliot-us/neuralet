import os, re
import time
import cv2 as cv

class Distancing:

    def __init__(self, config):
        self.config = config
        self.ui = None
        self.detector = None
        self.device = self.config.get_section_dict('Detector')['Device']
        self.running_video = False

        if self.device == 'Jetson':
            from libs.detectors.jetson.Detector import Detector
            self.detector = Detector(self.config)
        elif self.device == 'EdgeTPU':
            from libs.detectors.edgetpu.Detector import Detector
            self.detector = Detector(self.config)
        elif self.device == 'Dummy':
            self.detector = None

        self.image_size = [int(i) for i in self.config.get_section_dict('Detector')['ImageSize'].split(',')]
        
        if self.device != 'Dummy':
            print('Device is: ', self.device)
            print('Detector is: ', self.detector.name)
            print('image size: ', self.image_size)

    def set_ui(self, ui):
        self.ui = ui

    def __process(self, cv_image):

        if self.device == 'Dummy': 
            return cv_image, [], None

        resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        objects_list = self.detector.inference(rgb_resized_image)
        hscale = cv_image.shape[0]/resized_image.shape[0]
        wscale = cv_image.shape[1]/resized_image.shape[1]
        for i in range(len(objects_list)):
            box = objects_list[i]["bbox"]
            x0 = int(box[1] * cv_image.shape[1])
            y0 = int(box[0] * cv_image.shape[0])
            x1 = int(box[3] * cv_image.shape[1])
            y1 = int(box[2] * cv_image.shape[0])
            objects_list[i]["bbox"] = [x0, y0, x1 - x0, y1 - y0] 

        distancings = self.calculate_distancing(objects_list)
        return cv_image, objects_list, distancings

    def process_video(self, video_uri):
        self.running_video = True
        input_cap = cv.VideoCapture(video_uri)

        if (input_cap.isOpened()):
            print('opened video ', video_uri)
        else:
            print('failed to load video ', video_uri)
            return 

        while input_cap.isOpened() and self.running_video:
            _, cv_image = input_cap.read()
            _, objects, distancings = self.__process(cv_image)
            self.ui.update(cv_image, objects, distancings)
            time.sleep(0.030) 

        input_cap.release()
        self.running_video = False

    def process_image(self, image_path):
        cv_image = cv.imread(image_path)
        _, objects, distancings = self.__process(cv_image)
        self.ui.update(cv_image, objects, distancings) 

    def calculate_distancing(self, objects_list):
        pass
        

