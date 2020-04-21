import time
import cv2 as cv
import numpy as np
import math
from libs.centroid_object_tracker import CentroidTracker
from scipy.spatial import distance as dist


class Distancing:

    def __init__(self, config):
        self.config = config
        self.ui = None
        self.detector = None
        self.device = self.config.get_section_dict('Detector')['Device']
        self.running_video = False
        self.tracker = CentroidTracker(
            max_disappeared=int(self.config.get_section_dict("PostProcessor")["MaxTrackFrame"]))
        if self.device == 'Jetson':
            from libs.detectors.jetson.detector import Detector
            self.detector = Detector(self.config)
        elif self.device == 'EdgeTPU':
            from libs.detectors.edgetpu.detector import Detector
            self.detector = Detector(self.config)
        elif self.device == 'Dummy':
            self.detector = None

        self.image_size = [int(i) for i in self.config.get_section_dict('Detector')['ImageSize'].split(',')]

        if self.device != 'Dummy':
            print('Device is: ', self.device)
            print('Detector is: ', self.detector.name)
            print('image size: ', self.image_size)

        self.dist_method = self.config.get_section_dict("PostProcessor")["DistMethod"]
        self.dist_threshold = self.config.get_section_dict("PostProcessor")["DistThreshold"]

    def set_ui(self, ui):
        self.ui = ui

    def __process(self, cv_image):
        """
        return object_list list of  dict for each obj,
        obj["bbox"] is normalized coordinations for [x0, y0, x1, y1] of box
        """
        if self.device == 'Dummy':
            return cv_image, [], None

        # Resize input image to resolution
        resolution = [int(i) for i in self.config.get_section_dict('App')['Resolution'].split(',')]
        cv_image = cv.resize(cv_image, tuple(resolution))

        resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        tmp_objects_list = self.detector.inference(rgb_resized_image)
        [w,h] = resolution

        for obj in tmp_objects_list:
            box = obj["bbox"]
            x0 = box[1]
            y0 = box[0]
            x1 = box[3]
            y1 = box[2]
            obj["centroid"] = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
            obj["bbox"] = [x0, y0, x1, y1]
            obj["centroidReal"]=[(x0 + x1)*w / 2, (y0 + y1)*h / 2, (x1 - x0)*w, (y1 - y0)*h]
            obj["bboxReal"]=[x0*w,y0*h,x1*w,y1*h]
 
        objects_list, distancings = self.calculate_distancing(tmp_objects_list)
        return cv_image, objects_list, distancings

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
            cv_image, objects, distancings = self.__process(cv_image)

            self.ui.update(cv_image, objects, distancings)
        input_cap.release()
        self.running_video = False

    def process_image(self, image_path):
        # Process and pass the image to ui modules
        cv_image = cv.imread(image_path)
        cv_image, objects, distancings = self.__process(cv_image)
        self.ui.update(cv_image, objects, distancings)

    def calculate_distancing(self, objects_list):
        """
        this function post-process the raw boxes of object detector and calculate a distance matrix
        for detected bounding boxes.
        post processing is consist of:
        1. omitting large boxes by filtering boxes which are biger than the 1/4 of the size the image.
        2. omitting duplicated boxes by applying an auxilary non-maximum-suppression.
        3. apply a simple object tracker to make the detection more robust.

        params:
        object_list: a list of dictionaries. each dictionary has attributes of a detected object such as
        "id", "centroid" (a tuple of the normalized centroid coordinates (cx,cy,w,h) of the box) and "bbox" (a tuple
        of the normalized (xmin,ymin,xmax,ymax) coordinate of the box)

        returns:
        object_list: the post processed version of the input
        distances: a NxN ndarray which i,j element is distance between i-th and l-th bounding box

        """
        new_objects_list = self.ignore_large_boxes(objects_list)
        new_objects_list = self.non_max_suppression_fast(new_objects_list,
                                                         float(self.config.get_section_dict("PostProcessor")[
                                                                   "NMSThreshold"]))
        tracked_boxes = self.tracker.update(new_objects_list)
        new_objects_list = [tracked_boxes[i] for i in tracked_boxes.keys()]
        for i, item in enumerate(new_objects_list):
            item["id"] = item["id"].split("-")[0] + "-" + str(i)

        centroids = np.array( [obj["centroid"] for obj in new_objects_list] )
        distances = self.calculate_box_distances(new_objects_list)

        return new_objects_list, distances

    @staticmethod
    def ignore_large_boxes(object_list):

        """
        filtering boxes which are biger than the 1/4 of the size the image
        params:
            object_list: a list of dictionaries. each dictionary has attributes of a detected object such as
            "id", "centroid" (a tuple of the normalized centroid coordinates (cx,cy,w,h) of the box) and "bbox" (a tuple
            of the normalized (xmin,ymin,xmax,ymax) coordinate of the box)
        returns:
        object_list: input object list without large boxes
        """
        large_boxes = []
        for i in range(len(object_list)):
            if (object_list[i]["centroid"][2] * object_list[i]["centroid"][3]) > 0.25:
                large_boxes.append(i)
        updated_object_list = [j for i, j in enumerate(object_list) if i not in large_boxes]
        return updated_object_list

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


    def calculate_distance_of_two_points_of_boxes(self,first_point, second_point):
    
        """
        This function calculates a distance L for two input corresponding points of two detected bounding boxes.
        it is assumed that each person is H = 170 cm tall in real scene to map the distances in the image (in pixels) to 
        physical distance measures (in meters). 

        params:
        first_point: (x, y, h)-tuple, where x,y is the location of a point (center or each of 4 corners of a bounding box)
        and h is the height of the bounding box. 
        second_point: same tuple as first_point for the corresponding point of other box 

        returns:
        L:  Estimated physical distance (in centimeters) between first_point and second_point.


        """

        # estimate corresponding points distance
        [xc1, yc1, h1] = first_point
        [xc2, yc2, h2] = second_point
        
        Dx = xc2 - xc1
        Dy = yc2 - yc1
        
        Lx = Dx * 170 * (1/h1 + 1/h2)/2
        Ly = Dy * 170 * (1/h1 + 1/h2)/2
        
        L=math.sqrt(Lx**2+Ly**2)
        
        return L 


    def calculate_box_distances(self, nn_out):
        
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
            distanceRow=[]
            for j in range(len(nn_out)):
                if i == j:
                    L = 0
                else:
                    if ( int(self.dist_method) == 2 ):
                        first_point_of_first_box = [nn_out[i]["bboxReal"][0],nn_out[i]["bboxReal"][1],nn_out[i]["bboxReal"][3]-nn_out[i]["bboxReal"][1]]
                        second_point_of_first_box = [nn_out[i]["bboxReal"][2],nn_out[i]["bboxReal"][1],nn_out[i]["bboxReal"][3]-nn_out[i]["bboxReal"][1]]
                        third_point_of_first_box = [nn_out[i]["bboxReal"][0],nn_out[i]["bboxReal"][3],nn_out[i]["bboxReal"][3]-nn_out[i]["bboxReal"][1]]
                        forth_point_of_first_box = [nn_out[i]["bboxReal"][2],nn_out[i]["bboxReal"][3],nn_out[i]["bboxReal"][3]-nn_out[i]["bboxReal"][1]]
                        
                        first_point_of_second_box = [nn_out[j]["bboxReal"][0],nn_out[j]["bboxReal"][1],nn_out[j]["bboxReal"][3]-nn_out[j]["bboxReal"][1]]
                        second_point_of_second_box = [nn_out[j]["bboxReal"][2],nn_out[j]["bboxReal"][1],nn_out[j]["bboxReal"][3]-nn_out[j]["bboxReal"][1]]
                        third_point_of_second_box = [nn_out[j]["bboxReal"][0],nn_out[j]["bboxReal"][3],nn_out[j]["bboxReal"][3]-nn_out[j]["bboxReal"][1]]
                        forth_point_of_second_box = [nn_out[j]["bboxReal"][2],nn_out[j]["bboxReal"][3],nn_out[j]["bboxReal"][3]-nn_out[j]["bboxReal"][1]]

                        L1 = self.calculate_distance_of_two_points_of_boxes(first_point_of_first_box, first_point_of_second_box)
                        L2 = self.calculate_distance_of_two_points_of_boxes(second_point_of_first_box, second_point_of_second_box)
                        L3 = self.calculate_distance_of_two_points_of_boxes(third_point_of_first_box, third_point_of_second_box)
                        L4 = self.calculate_distance_of_two_points_of_boxes(forth_point_of_first_box, forth_point_of_second_box)
                        
                        L = min(L1, L2, L3, L4)
                    elif ( int(self.dist_method) == 1 ):
                        center_of_first_box = [nn_out[i]["centroidReal"][0],nn_out[i]["centroidReal"][1],nn_out[i]["centroidReal"][3]]
                        center_of_second_box = [nn_out[j]["centroidReal"][0],nn_out[j]["centroidReal"][1],nn_out[j]["centroidReal"][3]]

                        L = self.calculate_distance_of_two_points_of_boxes(center_of_first_box, center_of_second_box) 
                distanceRow.append(L)    
            distances.append(distanceRow)
        return distances 



