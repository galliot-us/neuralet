import time


class Logger:
    """logger layer to build a logger and pass data to it for logging

    this class build a layer based on config specification and call update
    method of it based on logging frequency

        :param config: a ConfigEngine object which store all of the config parameters. Access  to any parameter
        is possible by calling get_section_dict method.
        :param fps: frame rate of the video file, it should be specified by user in config file.
        :param name: logger name, at this time only csv_logger is supported. You can implement your own logger
        by following csv_logger implementation as an example.
        :param logger: Logger object which is built based on the name parameter specified in config file.
        :param time_interval: specifies how often the logger should log information. For example with time_interval of 0.5
        the logger log the information every 0.5 seconds. (set zero to log in every frame)
        :param frame_number: keep track of current frame number of the video. This will be used by update method to decide
        whether its time to log information or logging should be skipped for this frame (based on logger time interval.)


    """

    def __init__(self, config):
        """build the logger and initialize the frame number and set attributes"""
        self.config = config
        self.fps = int(self.config.get_section_dict("Logger")["Fps"])
        self.name = self.config.get_section_dict("Logger")["Name"]
        if self.name == "csv_logger":
            from . import csv_logger
            self.logger = csv_logger.Logger(self.config)
        self.time_interval = float(self.config.get_section_dict("Logger")["TimeInterval"])  # Seconds
        self.submited_time = 0

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
