import time


class Logger:
    """logger layer to build a logger and pass data to it for logging

    this class build a layer based on config specification and call update
    method of it based on logging frequency

        :param config: a ConfigEngine object which store all of the config parameters. Access  to any parameter
        is possible by calling get_section_dict method.
    """

    def __init__(self, config):
        """build the logger and initialize the frame number and set attributes"""
        self.config = config
        # Logger name, at this time only csv_logger is supported. You can implement your own logger
        # by following csv_logger implementation as an example.
        self.name = self.config.get_section_dict("Logger")["Name"]
        if self.name == "csv_logger":
            from . import csv_processed_logger
            self.logger = csv_processed_logger.Logger(self.config)

            # For Logger instance from loggers/csv_logger
            # region csv_logger
            # from . import csv_logger
            # self.logger = csv_logger.Logger(self.config)
            # end region

        # Specifies how often the logger should log information. For example with time_interval of 0.5
        # the logger log the information every 0.5 seconds.
        self.time_interval = float(self.config.get_section_dict("Logger")["TimeInterval"])  # Seconds
        self.submited_time = 0
        # self.frame_number = 0  # For Logger instance from loggers/csv_logger

    def update(self, objects_list, distances):
        """call the update method of the logger.

        based on frame_number, fps and time interval, it decides whether to call the
        logger's update method to store the data or not.

        Args:
            objects_list: a list of dictionary where each dictionary stores information of an object (person) in a frame.
            distances: a 2-d numpy array that stores distance between each pair of objects.
        """

        if time.time() - self.submited_time > self.time_interval:
            self.logger.update(objects_list, distances)
            self.submited_time = time.time()
            # For Logger instance from loggers/csv_logger
            # region
            # self.logger.update(self.frame_number, objects_list, distances)
            # self.frame_number += 1
            # end region

