import os, re
import time

import numpy as np
import cv2 as cv
from scipy.spatial import distance as dist
from libs.centroid_object_tracker import CentroidTracker

class Distancing:

    def __init__(self, config):
        self.config = config
        self.ui = None
        self.detector = None
        self.device = self.config.get_section_dict('Detector')['Device']
        self.running_video = False
        self.tracker = CentroidTracker(maxDisappeared=5)
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
        """
        return object_list list of  dict for each obj,
        obj["bbox"] is normalized coordinations for [x0, y0, x1, y1] of box
        """
        if self.device == 'Dummy':
            return cv_image, [], None

        resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        tmp_objects_list = self.detector.inference(rgb_resized_image)
        hscale = cv_image.shape[0]/resized_image.shape[0]
        wscale = cv_image.shape[1]/resized_image.shape[1]

        for obj in tmp_objects_list:
            box = obj["bbox"]
            x0 = box[1]
            y0 = box[0]
            x1 = box[3]
            y1 = box[2]
            obj["centroid"] = [(x0+x1)/2, (y0+y1)/2, x1 - x0, y1 - y0]
            obj["bbox"] = [x0, y0, x1, y1]

        objects_list, distancings = self.calculate_distancing(tmp_objects_list)
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

        input_cap.release()
        self.running_video = False

    def process_image(self, image_path):
        cv_image = cv.imread(image_path)
        _, objects, distancings = self.__process(cv_image)
        self.ui.update(cv_image, objects, distancings)

    def calculate_distancing(self, objects_list):
        new_objects_list = self.ignore_large_boxes(objects_list)
        new_objects_list = self.non_max_suppression_fast(new_objects_list, 0.98)
        tracked_boxes = self.tracker.update(new_objects_list)
        new_objects_list = [tracked_boxes[i] for i in tracked_boxes.keys()]
        for i, item in enumerate(new_objects_list):
            item["id"] = item["id"].split("-")[0] + "-" + str(i)

        centroids = np.array( [obj["centroid"] for obj in new_objects_list] )
        distances = dist.cdist(centroids, centroids)
        return new_objects_list, distances

    @staticmethod
    def ignore_large_boxes(object_list):
        large_boxes = []
        for i in range(len(object_list)):
            if (object_list[i]["centroid"][2] * object_list[i]["centroid"][3]) > 0.25:
                large_boxes.append(i)
        updated_object_list = [j for i,j in enumerate(object_list) if i not in large_boxes]
        return updated_object_list

    @staticmethod
    def non_max_suppression_fast(object_list, overlapThresh):
        # if there are no boxes, return an empty list
        boxes = np.array([item["centroid"] for item in object_list])
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        cy = boxes[:,1]
        cx = boxes[:,0]
        h = boxes[:,3]
        w = boxes[:,2]
        x1 = cx - (w/2)
        x2 = cx + (w/2)
        y1 = cy - (h/2)
        y2 = cy + (h/2)
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (h + 1) * (w + 1)
        idxs = np.argsort(cy + (h/2))
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        #integer data
        updated_object_list = [j for i,j in enumerate(object_list) if i in pick]
        return updated_object_list

