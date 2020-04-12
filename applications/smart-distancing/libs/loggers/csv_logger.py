import csv
import os
from datetime import date

import numpy as np


class Logger:
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
        file_name = str(date.today())
        objects_log_file_path = os.path.join(self.objects_log_directory, file_name + ".csv")
        distances_log_file_path = os.path.join(self.distances_log_directory, file_name + ".csv")
        self.log_objects(object_list, frame_number, objects_log_file_path)
        self.log_distances(distances, frame_number, distances_log_file_path)

    @staticmethod
    def log_objects(object_list, frame_number, file_path):
        for object_item in object_list:
            object_dict = {}
            object_dict.update({"frame_number": frame_number})
            for key, value in object_item.items():
                if isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        object_dict.update({str(key) + "_" + str(i): item})
                else:
                    object_dict.update({key: value})

            if os.path.exists(file_path):
                with open(file_path, "a", newline="") as csvfile:
                    field_names = list(object_dict.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writerow(object_dict)
            else:
                with open(file_path, "w", newline="") as csvfile:
                    field_names = list(object_dict.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                    writer.writerow(object_dict)

    def log_distances(self, distances, frame_number, file_path):
        violating_objects = self.extract_violating_objects(distances)
        for indices in violating_objects:
            if os.path.exists(file_path):
                with open(file_path, "a", newline="") as csvfile:
                    field_names = ["frame_number", "object_0", "object_1", "distance"]
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writerow({"frame_number": frame_number,
                                     "object_0": indices[0],
                                     "object_1": indices[1],
                                     "distance": distances[indices[0], indices[1]]})
            else:
                with open(file_path, "w", newline="") as csvfile:
                    field_names = ["frame_number", "object_0", "object_1", "distance"]
                    writer = csv.DictWriter(csvfile, fieldnames=field_names)
                    writer.writeheader()
                    writer.writerow({"frame_number": frame_number,
                                     "object_0": indices[0],
                                     "object_1": indices[1],
                                     "distance": distances[indices[0], indices[1]]})

    def extract_violating_objects(self, distances):
        triu_distances = np.triu(distances) + np.tril(10 * np.ones(distances.shape))
        violating_objects = np.argwhere(
            triu_distances < float(self.config.get_section_dict("Detector")["DistThreshold"]))
        return violating_objects
