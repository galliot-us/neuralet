class Logger:
    def __init__(self, config):
        self.config = config
        self.fps = int(self.config.get_section_dict("Logger")["Fps"])
        self.name = self.config.get_section_dict("Logger")["Name"]
        if self.name == "csv_logger":
            from . import csv_logger
            self.logger = csv_logger.Logger(self.config)
        self.time_interval = float(self.config.get_section_dict("Logger")["TimeInterval"])
        self.frame_number = 0

    def update(self, object_list, distances):
        if self.frame_number % int(self.fps * self.time_interval) == 0:
            self.logger.update(self.frame_number, object_list, distances)
            self.frame_number += 1
        else:
            self.frame_number += 1
