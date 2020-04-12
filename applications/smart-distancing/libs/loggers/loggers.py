class Logger:
    def __init__(self, fps, config):
        self.fps = fps
        self.config = config
        self.name = self.config.get_section_dict("Logger")["name"]
        if self.name == "csv_logger":
            from . import csv_logger
            self.logger = csv_logger.Logger()
        self.time_interval = self.config.get_section_dict("Logger")["time_interval"]
        self.frame_number = 0

    def update(self, object_list, distances):
        if self.frame_number % int(self.fps * self.time_interval) == 0:
            self.logger.update(object_list, distances)
            self.frame_number += 1
        else:
            self.frame_number += 1
