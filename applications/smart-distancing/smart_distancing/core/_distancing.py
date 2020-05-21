"""Distancing base class and common functions."""
import abc
import math
import logging

from typing import (
    Dict,
    Tuple,
)

import numpy as np

import smart_distancing as sd

__all__ = ['BaseDistancing']

logger = logging.getLogger(__name__)

class BaseDistancing(abc.ABC):
    """Distancing base class"""

    config = None  # type: sd.core.ConfigEngine
    ui = None  # type: sd.ui.WebGUI
    detector = None  # type: sd.detectors.BaseDetector
    running_video = False
    tracker = None

    def __init__(self, config):
        # set up the object's logger
        logger.debug('__init__ start')

        # assign the config
        self.config = config

        # log the device type
        logger.info(f'device is: {self.device}')
        logger.info(f'image size: {self.image_size}')

        # set the output resolution
        # this is here rather than as a property becuase it's accessed repeatedly
        # although a cached property could be implemented instead.
        self.resolution = tuple(int(i) for i in self.distancing_config['Resolution'].split(','))
        logger.debug(f'{self.__class__.__name__}:resolution set to:{self.resolution}')

        # create the ui
        # there is a self reference issue here that could become
        # a problem if the engine needs to be restarted repeatedly
        self.ui = sd.ui.WebGUI(self.config, self)

    @property
    def distancing_config(self) -> Dict:
        return self.config.get_section_dict('App')

    @property
    def detector_config(self) -> Dict:
        """:return: the Detector section from the .ini config"""
        return self.config.get_section_dict('Detector')

    @property
    def post_processor_config(self) -> Dict:
        """:return: the PostProcessor section from the .ini config"""
        return self.config.get_section_dict("PostProcessor")

    @property
    def dist_method(self):
        return self.post_processor_config["DistMethod"]

    @property
    def dist_threshold(self):
        return self.post_processor_config["DistThreshold"]

    @property
    def device(self) -> str:
        return self.detector_config['Device']

    @property
    def image_size(self) -> Tuple[int, int]:
        return tuple(int(i) for i in self.detector_config['ImageSize'].split(','))

    @abc.abstractmethod
    def process_video(self, video_path:str):
        """process video and update the ui"""

    @staticmethod
    def ignore_large_boxes(objects_dict, thresh=0.25):

        """
        Filter boxes which are bigger than 1/4 of the size the image.

        Args:
            object_list (:obj:`dict` of :obj:`dict`): Each value has attributes of a
                detected object such as "id", "centroid" (a tuple of the normalized centroid
                coordinates (cx,cy,w,h) of the box) and "bbox" (a tuple of the normalized 
                (xmin,ymin,xmax,ymax) coordinate of the box). Each key is an object uid.
            thresh:
                size threshold
        Returns:
            object_dict (:obj:`dict` of :obj:`dict`):
                a dict with the same items, but only if:
                (record["centroid"][2] * record["centroid"][3]) > thresh
        """
        return {
            uid: record for uid, record in objects_dict.items()
            if (record["centroid"][2] * record["centroid"][3]) > thresh
        }

    @staticmethod
    def non_max_suppression_fast(object_list, overlapThresh):

        """
        omitting duplicated boxes by applying an auxilary non-maximum-suppression.
        params:
        object_list: a list of dictionaries. each dictionary has attributes of a detected object such
        "id", "centroid" (a tuple of the normalized centroid coordinates (cx,cy,w,h) of the box) and "bbox" (a tuple
        of the normalized (xmin,ymin,xmax,ymax) coordinate of the box)

        overlapThresh: threshold of minimum IoU of to detect two box as duplicated.

        returns:
        object_list: input object list without duplicated boxes
        """
        # TODO(mdegans?): rewrite to use dict 
        # if there are no boxes, return an empty list
        boxes = np.array([item["centroid"] for item in object_list])
        corners = np.array([item["bbox"] for item in object_list])
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes
        pick = []
        cy = boxes[:, 1]
        cx = boxes[:, 0]
        h = boxes[:, 3]
        w = boxes[:, 2]
        x1 = corners[:, 0]
        x2 = corners[:, 2]
        y1 = corners[:, 1]
        y2 = corners[:, 3]
        area = (h + 1) * (w + 1)
        idxs = np.argsort(cy + (h / 2))
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        updated_object_list = [j for i, j in enumerate(object_list) if i in pick]
        return updated_object_list

    @staticmethod
    def calculate_object_distance(first_object:dict, second_object:dict):

        """
        This function calculates a distance l for two input corresponding points of two detected bounding boxes.
        it is assumed that each person is H = 170 cm tall in real scene to map the distances in the image (in pixels) to
        physical distance measures (in meters).

        params:
        first_point: (x, y, h)-tuple, where x,y is the location of a point (center or each of 4 corners of a bounding box)
        and h is the height of the bounding box.
        second_point: same tuple as first_point for the corresponding point of other box

        returns:
        l:  Estimated physical distance (in centimeters) between first_point and second_point.


        """

        # estimate corresponding points distance
        [xc1, yc1, h1] = first_point
        [xc2, yc2, h2] = second_point

        dx = xc2 - xc1
        dy = yc2 - yc1

        lx = dx * 170 * (1 / h1 + 1 / h2) / 2
        ly = dy * 170 * (1 / h1 + 1 / h2) / 2

        l = math.sqrt(lx ** 2 + ly ** 2)

        return l

    def calculate_object_distances(self, nn_out):

        """
        This function calculates a distance matrix for detected bounding boxes.
        Two methods are implemented to calculate the distances, first one estimates distance of center points of the
        boxes and second one uses minimum distance of each of 4 points of bounding boxes.

        params:
        object_list: a list of dictionaries. each dictionary has attributes of a detected object such as
        "id", "centroidReal" (a tuple of the centroid coordinates (cx,cy,w,h) of the box) and "bboxReal" (a tuple
        of the (xmin,ymin,xmax,ymax) coordinate of the box)

        returns:
        distances: a NxN ndarray which i,j element is estimated distance between i-th and j-th bounding box in real scene (cm)

        """

        distances = []
        for i in range(len(nn_out)):
            distance_row = []
            for j in range(len(nn_out)):
                if i == j:
                    l = 0
                else:
                    if (self.dist_method == 'FourCornerPointsDistance'):
                        lower_left_of_first_box = [nn_out[i]["bboxReal"][0], nn_out[i]["bboxReal"][1],
                                                   nn_out[i]["centroidReal"][3]]
                        lower_right_of_first_box = [nn_out[i]["bboxReal"][2], nn_out[i]["bboxReal"][1],
                                                    nn_out[i]["centroidReal"][3]]
                        upper_left_of_first_box = [nn_out[i]["bboxReal"][0], nn_out[i]["bboxReal"][3],
                                                   nn_out[i]["centroidReal"][3]]
                        upper_right_of_first_box = [nn_out[i]["bboxReal"][2], nn_out[i]["bboxReal"][3],
                                                    nn_out[i]["centroidReal"][3]]

                        lower_left_of_second_box = [nn_out[j]["bboxReal"][0], nn_out[j]["bboxReal"][1],
                                                    nn_out[j]["centroidReal"][3]]
                        lower_right_of_second_box = [nn_out[j]["bboxReal"][2], nn_out[j]["bboxReal"][1],
                                                     nn_out[j]["centroidReal"][3]]
                        upper_left_of_second_box = [nn_out[j]["bboxReal"][0], nn_out[j]["bboxReal"][3],
                                                    nn_out[j]["centroidReal"][3]]
                        upper_right_of_second_box = [nn_out[j]["bboxReal"][2], nn_out[j]["bboxReal"][3],
                                                     nn_out[j]["centroidReal"][3]]

                        l1 = self.calculate_distance_of_two_points_of_boxes(lower_left_of_first_box,
                                                                            lower_left_of_second_box)
                        l2 = self.calculate_distance_of_two_points_of_boxes(lower_right_of_first_box,
                                                                            lower_right_of_second_box)
                        l3 = self.calculate_distance_of_two_points_of_boxes(upper_left_of_first_box,
                                                                            upper_left_of_second_box)
                        l4 = self.calculate_distance_of_two_points_of_boxes(upper_right_of_first_box,
                                                                            upper_right_of_second_box)

                        l = min(l1, l2, l3, l4)
                    elif (self.dist_method == 'CenterPointsDistance'):
                        center_of_first_box = [nn_out[i]["centroidReal"][0], nn_out[i]["centroidReal"][1],
                                               nn_out[i]["centroidReal"][3]]
                        center_of_second_box = [nn_out[j]["centroidReal"][0], nn_out[j]["centroidReal"][1],
                                                nn_out[j]["centroidReal"][3]]

                        l = self.calculate_distance_of_two_points_of_boxes(center_of_first_box, center_of_second_box)
                distance_row.append(l)
            distances.append(distance_row)
        distances_asarray = np.asarray(distances, dtype=np.float32)
        return distances_asarray


    def calculate_distancing(self, objects:dict):
        """
        this function post-process the raw boxes of object detector and calculate a distance matrix
        for detected bounding boxes.
        post processing is consist of:
        1. omitting large boxes by filtering boxes which are biger than the 1/4 of the size the image.
        2. omitting duplicated boxes by applying an auxilary non-maximum-suppression.
        3. apply a simple object tracker to make the detection more robust.

        params:
        object_list: a dict of dictionaries. each item has attributes of a detected object such as
        "id", "centroid" (a tuple of the normalized centroid coordinates (cx,cy,w,h) of the box) and "bbox" (a tuple
        of the normalized (xmin,ymin,xmax,ymax) coordinate of the box) each key is a bbox unique id or
        other unique integer.

        returns:
        object_list: the post processed version of the input
        distances: a NxN ndarray which i,j element is distance between i-th and l-th bounding box

        """
        objects = self.ignore_large_boxes(objects)
        # we can skip this if we already have a uid for each object (eg. DsEngine)
        if self.tracker:
            overlap_thresh = float(self.config.get_section_dict("PostProcessor")["NMSThreshold"])
            objects = self.non_max_suppression_fast(objects, overlap_thresh)
            objects = self.tracker.update(objects)

        objects = self.calculate_box_distances(objects)

        return objects
