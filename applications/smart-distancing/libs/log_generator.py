import os


class Logger:
    def __init__(self, fps, time_interval, file_name):
        self.fps = fps
        self.time_interval = time_interval
        self.frame_number = 0
        self.log_file = os.path.join("/logs", file_name)
        with open(self.log_file):
            pass

    def update(self, object_list, distances):
        if self.frame_number % int(self.fps * self.time_interval) == 0:
            self.frame_number += 1
            pass
        else:
            self.frame_number += 1
