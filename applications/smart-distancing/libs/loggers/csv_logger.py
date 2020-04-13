import csv
import os
from datetime import date

import numpy as np


def prepare_object(object_item, frame_number):
    """construct a dictionary that is appropriate for csv writer.

    this function transform a dictionary with list values to a dictionary
    with scalar values. This transformation is necessary for csv writer to avoid
    writing lists into csv.

    Args:
        object_item: it is a dictionary that contains an detected object information after postprocessing
        frame_number: current frame number

    Returns:
        A transformed version of object_item to a dictionary with only scalar values. It also contains an item
        for frame number.

    """
    object_dict = {}
    object_dict.update({"frame_number": frame_number})
    for key, value in object_item.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, float):
                    item = round(item, 4)
                elif isinstance(item, np.float32):
                    item = round(float(item), 4)

                object_dict.update({str(key) + "_" + str(i): item})
        else:
            if isinstance(value, float):
                value = round(value, 4)
            elif isinstance(value, np.float32):
                value = round(float(value), 4)
            object_dict.update({key: value})
    return object_dict


class Logger:
    """A CSV logger class that store objects information and violated distances information into csv files.

    this logger creates two csv file every day in two different directory, one for logging detected objects
    and one for logging violated social distancing incidents. the file names are the same as recording date.

    Attributes:
        config: a ConfigEngine object which store all of the config parameters. access  to any parameter
        is possible by calling get_section_dict method.
        log_directory: the parent directory that stores all log file.
        objects_log_directory: a directory inside the log_directory that stores object log files.
        distances_log_directory: a directory inside the log_directory that stores violated distances log files.


    """

    def __init__(self, config):
        self.config = config
        self.log_directory = config.get_section_dict("Logger")["LogDirectory"]
        self.objects_log_directory = os.path.join(self.log_directory, "objects_log")
        self.distances_log_directory = os.path.join(self.log_directory, "distances_log")
        if not os.path.exists(self.log_directory):
            os.mkdir(self.log_directory)
        if not os.path.exists(self.objects_log_directory):
            os.mkdir(self.objects_log_directory)
        if not os.path.exists(self.distances_log_directory):
            os.mkdir(self.distances_log_directory)

    def update(self, frame_number, object_list, distances):
        """write the object and violated distances information of a frame into log files.

        Args:
            frame_number: current frame number
            object_list: a list of dictionary where each dictionary stores information of an object (person) in a frame.
            distances: a 2-d numpy array that stores distance between each pair of objects.
        """
        file_name = str(date.today())
        objects_log_file_path = os.path.join(self.objects_log_directory, file_name + ".csv")
        distances_log_file_path = os.path.join(self.distances_log_directory, file_name + ".csv")
        self.log_objects(object_list, frame_number, objects_log_file_path)
        self.log_distances(distances, frame_number, distances_log_file_path)

    @staticmethod
    def log_objects(object_list, frame_number, file_path):
        """write objects information of a frame into the object log file.
        each row of the object log file consist of a detected object (person) information such as
        object (person) ids, bounding box coordinates and frame number.

        Args:
            object_list: a list of dictionary where each dictionary stores information of an object (person) in a frame.
            frame_number: current frame number
            file_path: log file path
        """
        if len(object_list) != 0:
            object_dict = list(map(lambda x: prepare_object(x, frame_number), object_list))

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
        """write violated incident's information of a frame into the object log file

        each row of the distances log file consist of a violation information such as object (person) ids,
        distance between these two object and frame number.

        Args:
            distances: a 2-d numpy array that stores distance between each pair of objects.
            frame_number: current frame number
            file_path: log file path
        """
        violating_objects = self.extract_violating_objects(distances)
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

    def extract_violating_objects(self, distances):
        """extract pair of objects that are closer than the distance threshold.

        Args:
            distances: a 2-d numpy array that stores distance between each pair of objects.

        Returns:
            A 2-d numpy array where each row is the ids of the objects that violated the social distancing.

        """
        triu_distances = np.triu(distances) + np.tril(10 * np.ones(distances.shape))
        violating_objects = np.argwhere(
            triu_distances < float(self.config.get_section_dict("Detector")["DistThreshold"]))
        return violating_objects
