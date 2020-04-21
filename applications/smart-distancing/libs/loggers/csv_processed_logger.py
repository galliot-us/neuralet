import csv
import os
from datetime import date, datetime
from libs.tools.environment_score import mx_environment_scoring_consider_crowd

import numpy as np


class Logger:
    """A CSV logger class that store objects information and violated distances information into csv files.

    This logger creates two csv file every day in two different directory, one for logging detected objects
    and violated social distancing incidents. The file names are the same as recording date.

    :param config: A ConfigEngine object which store all of the config parameters. Access to any parameter
        is possible by calling get_section_dict method.
    """

    def __init__(self, config):
        self.config = config
        # The parent directory that stores all log file.
        self.log_directory = config.get_section_dict("Logger")["LogDirectory"]
        # A directory inside the log_directory that stores object log files.
        self.objects_log_directory = os.path.join(self.log_directory, "objects_log")

        if not os.path.exists(self.log_directory):
            os.mkdir(self.log_directory)

        if not os.path.exists(self.objects_log_directory):
            os.mkdir(self.objects_log_directory)

    def update(self, objects_list, distances):
        """Write the object and violated distances information of a frame into log files.

        Args:
            objects_list: List of dictionary where each dictionary stores information of an object (person) in a frame.
            Distances: A 2-d numpy array that stores distance between each
        pair of objects.
            distances: A 2-d numpy array that stores distance between each pair of objects.
        """
        file_name = str(date.today())
        objects_log_file_path = os.path.join(self.objects_log_directory, file_name + ".csv")
        self.log_objects(objects_list, distances, objects_log_file_path)

    def log_objects(self, objects_list, distances, file_path):
        """Write objects information of a frame into the object log file.
        Each row of the object log file consist of a detected object (person) information such as
        object (person) ids, bounding box coordinates and frame number.

        Args:
            objects_list: A list of dictionary where each dictionary stores information of an object (person) in a frame.
            distances: A 2-d numpy array that stores distance between each pair of objects.
            file_path: The path for storing log files

        """

        violating_objects = self.extract_violating_objects(distances)
        # Get the number of violating objects (people)
        no_violating_objects = len(violating_objects)
        # Get the number of detected objects (people)
        no_detected_objects = len(objects_list)
        # Get environment score
        environment_score = mx_environment_scoring_consider_crowd(no_detected_objects, no_violating_objects)
        # Get timeline which is used for as Timestamp
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a") as csvfile:
            headers = ["Timestamp", "DetectedObjects", "ViolatingObjects", "EnvironmentScore"]
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {'Timestamp': current_time, 'DetectedObjects': no_detected_objects,
                 'ViolatingObjects': no_violating_objects, 'EnvironmentScore': environment_score})

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
