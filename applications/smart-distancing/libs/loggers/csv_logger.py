import csv
import os
from datetime import date
from tools.objects_post_process import extract_violating_objects

import numpy as np


def prepare_object(detected_object, frame_number):
    """Construct a dictionary that is appropriate for csv writer.

    This function transform a dictionary with list values to a dictionary
    with scalar values. This transformation is necessary for csv writer to avoid
    writing lists into csv.

    Args:
        detected_object: It is a dictionary that contains an detected object information after postprocessing.
        frame_number: current frame number

    Returns:
        A transformed version of detected_object to a dictionary with only scalar values. It also contains an item
        for frame number.

    """
    object_dict = {}
    object_dict.update({"frame_number": frame_number})
    for key, value in detected_object.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                # TODO: Inspect why some items are float and some are np.float32
                if isinstance(item, (float, np.float32)):
                    item = round(float(item), 4)
                object_dict.update({str(key) + "_" + str(i): item})
        else:
            # TODO: Inspect why some items are float and some are np.float32
            if isinstance(value, (float, np.float32)):
                value = round(float(value), 4)
            object_dict.update({key: value})
    return object_dict


class Logger:
    """A CSV logger class that store objects information and violated distances information into csv files.

    This logger creates two csv file every day in two different directory, one for logging detected objects
    and one for logging violated social distancing incidents. The file names are the same as recording date.

    :param config: A ConfigEngine object which store all of the config parameters. Access to any parameter
        is possible by calling get_section_dict method.
    """

    def __init__(self, config):
        self.config = config
        # The parent directory that stores all log file.
        self.log_directory = config.get_section_dict("Logger")["LogDirectory"]
        # A directory inside the log_directory that stores object log files.
        self.objects_log_directory = os.path.join(self.log_directory, "objects_log")
        self.distances_log_directory = os.path.join(self.log_directory, "distances_log")
        self.dist_threshold = config.get_section_dict("Detector")["DistThreshold"]
        if not os.path.exists(self.log_directory):
            os.mkdir(self.log_directory)
        if not os.path.exists(self.objects_log_directory):
            os.mkdir(self.objects_log_directory)
        if not os.path.exists(self.distances_log_directory):
            os.mkdir(self.distances_log_directory)

    def update(self, frame_number, objects_list, distances):
        """Write the object and violated distances information of a frame into log files.

        Args: frame_number: current frame number objects_list: A list of dictionary where each dictionary stores
        information of an object (person) in a frame. distances: A 2-d numpy array that stores distance between each
        pair of objects.
        """
        file_name = str(date.today())
        objects_log_file_path = os.path.join(self.objects_log_directory, file_name + ".csv")
        distances_log_file_path = os.path.join(self.distances_log_directory, file_name + ".csv")
        self.log_objects(objects_list, frame_number, objects_log_file_path)
        self.log_distances(distances, frame_number, distances_log_file_path)

    @staticmethod
    def log_objects(objects_list, frame_number, file_path):
        """Write objects information of a frame into the object log file.
        Each row of the object log file consist of a detected object (person) information such as
        object (person) ids, bounding box coordinates and frame number.

        Args: objects_list: A list of dictionary where each dictionary stores information of an object (person) in a
        frame. frame_number: current frame number file_path: log file path
        """
        if len(objects_list) != 0:
            object_dict = list(map(lambda x: prepare_object(x, frame_number), objects_list))

            if not os.path.exists(file_path):
                with open(file_path, "w", newline="") as csvfile:
                    field_names = list(object_dict[0].keys())
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()

            with open(file_path, "a", newline="") as csvfile:
                field_names = list(object_dict[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writerows(object_dict)

    def log_distances(self, distances, frame_number, file_path):
        """Write violated incident's information of a frame into the object log file.

        Each row of the distances log file consist of a violation information such as object (person) ids,
        distance between these two object and frame number.

        Args:
            distances: A 2-d numpy array that stores distance between each pair of objects.
            frame_number: current frame number
            file_path: The path for storing log files
        """
        violating_objects = extract_violating_objects(distances, self.dist_threshold)
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as csvfile:
                field_names = ["frame_number", "object_0", "object_1", "distance"]
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
        with open(file_path, "a", newline="") as csvfile:
            field_names = ["frame_number", "object_0", "object_1", "distance"]
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writerows([{"frame_number": frame_number,
                               "object_0": indices[0],
                               "object_1": indices[1],
                               "distance": distances[indices[0], indices[1]]} for indices in violating_objects])
