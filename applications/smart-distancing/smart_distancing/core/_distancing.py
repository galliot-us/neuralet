"""Distancing base class and common functions."""
import abc
import math
import logging

from typing import (
    Dict,
    Tuple,
    Optional,
)

import numpy as np

import smart_distancing as sd

__all__ = ['BaseDistancing']

logger = logging.getLogger(__name__)

class BaseDistancing(abc.ABC):
    """Distancing base class"""

    config = None  # type: sd.core.ConfigEngine
    ui = None  # type: sd.ui.WebGUI
    detector = None  # type: sd.detectors.BaseDetector
    running_video = False
    tracker = None

    def __init__(self, config):
        # set up the object's logger
        logger.debug('__init__ start')

        # assign the config
        self.config = config

        # log the device type
        logger.info(f'device is: {self.device}')
        logger.info(f'image size: {self.image_size}')

        # set the output resolution
        # this is here rather than as a property becuase it's accessed repeatedly
        # although a cached property could be implemented instead.
        self.resolution = tuple(int(i) for i in self.distancing_config['Resolution'].split(','))
        logger.debug(f'{self.__class__.__name__}:resolution set to:{self.resolution}')

        # create the ui
        # there is a self reference issue here that could become
        # a problem if the engine needs to be restarted repeatedly
        self.ui = sd.ui.WebGUI(self.config, self)

    @property
    def distancing_config(self) -> Dict:
        return self.config.get_section_dict('App')

    @property
    def detector_config(self) -> Dict:
        """:return: the Detector section from the .ini config"""
        return self.config.get_section_dict('Detector')

    @property
    def post_processor_config(self) -> Dict:
        """:return: the PostProcessor section from the .ini config"""
        return self.config.get_section_dict("PostProcessor")

    @property
    def dist_method(self):
        return self.post_processor_config["DistMethod"]

    @property
    def dist_threshold(self):
        return self.post_processor_config["DistThreshold"]

    @property
    def device(self) -> str:
        return self.detector_config['Device']

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        try:
            return tuple(int(i) for i in self.detector_config['ImageSize'].split(','))
        except KeyError:
            return None

    @abc.abstractmethod
    def process_video(self, video_path:str):
        """process video and update the ui"""

    def calculate_distancing(self, objects:dict):
        """
        todo(Mohsen?): implementation
        """
        pass
        # objects = self.ignore_large_boxes(objects)
        # # we can skip this if we already have a uid for each object (eg. DsEngine)
        # if self.tracker:
        #     overlap_thresh = float(self.config.get_section_dict("PostProcessor")["NMSThreshold"])
        #     objects = self.non_max_suppression_fast(objects, overlap_thresh)
        #     objects = self.tracker.update(objects)

        # objects = self.calculate_box_distances(objects)

        # return objects
