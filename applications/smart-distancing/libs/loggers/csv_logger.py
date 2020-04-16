import csv
import os
from datetime import date, datetime

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
        :param log_directory: The parent directory that stores all log file.
        :param objects_log_directory: A directory inside the log_directory that stores object log files.
        :param distances_log_directory: A directory inside the log_directory that stores violated distances log files.


    """

    def __init__(self, config):
        self.config = config
        self.log_directory = config.get_section_dict("Logger")["LogDirectory"]
        self.objects_log_directory = os.path.join(self.log_directory, "objects_log")
        # self.distances_log_directory = os.path.join(self.log_directory, "distances_log")

        if not os.path.exists(self.objects_log_directory):
            os.mkdir(self.objects_log_directory)
        # if not os.path.exists(self.distances_log_directory):
        #     os.mkdir(self.distances_log_directory)

    def update(self,objects_list, distances):
        """Write the object and violated distances information of a frame into log files.

        Args: frame_number: current frame number objects_list: A list of dictionary where each dictionary stores
        information of an object (person) in a frame. distances: A 2-d numpy array that stores distance between each
        pair of objects.
        """
        file_name = str(date.today())
        objects_log_file_path = os.path.join(self.objects_log_directory, file_name + ".csv")
        # distances_log_file_path = os.path.join(self.distances_log_directory, file_name + ".csv")
        self.log_objects(objects_list, distances, objects_log_file_path)
        # self.log_distances(distances, frame_number, distances_log_file_path)

    def log_objects(self, objects_list, distances, file_path):
        """Write objects information of a frame into the object log file.
        Each row of the object log file consist of a detected object (person) information such as
        object (person) ids, bounding box coordinates and frame number.

        Args:
            objects_list: A list of dictionary where each dictionary stores information of an object (person) in a frame.
            frame_number: Current frame number file_path: log file path.
            distances: A 2-d numpy array that stores distance between each pair of objects.

        """

        violating_objects = self.extract_violating_objects(distances)
        # Get timeline which is used for as Timestamp
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a") as csvfile:
            headers = ["Timestamp", "DetectedObjects", "ViolatingObjects"]
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {'Timestamp': current_time, 'DetectedObjects': len(objects_list),
                 'ViolatingObjects': len(violating_objects)})

    def log_distances(self, distances, frame_number, file_path):
        """Write violated incident's information of a frame into the object log file.

        Each row of the distances log file consist of a violation information such as object (person) ids,
        distance between these two object and frame number.

        Args:
            distances: A 2-d numpy array that stores distance between each pair of objects.
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
        """Extract pair of objects that are closer than the distance threshold.

        Args:
            distances: A 2-d numpy array that stores distance between each pair of objects.

        Returns:
            violating_objects: A 2-d numpy array where each row is the ids of the objects that violated the social distancing.

        """
        triu_distances = np.triu(distances) + np.tril(10 * np.ones(distances.shape))
        violating_objects = np.argwhere(
            triu_distances < float(self.config.get_section_dict("Detector")["DistThreshold"]))
        return violating_objects
