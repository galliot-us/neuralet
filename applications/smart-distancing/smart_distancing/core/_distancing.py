"""Distancing base class and common functions."""
import abc
import logging

from typing import (
    Dict,
)

__all__ = ['BaseDistancing']

class BaseDistancing(abc.ABC):
    """Distancing base class"""

    config = None  # type: sd.core.ConfigEngine
    ui = None  # type: sd.ui.WebGUI
    detector = None  # type: sd.detectors.BaseDetector

    def __init__(self, config):
        # set up the object's logger
        self.logger = logging.getLogger(self.__name__)
        self.logger.debug('__init__ start')

        # assign the config
        self.config = config

        # log the device type
        if self.device != 'Dummy':
            self.logger.info(f'Device is: {self.device}')
            self.logger.info(f'Detector is: {self.detector.name}')
            self.logger.info(f'image size: {self.image_size}')

        # create the ui
        # there is a self reference issue here that could become
        # a problem if the engine needs to be restarted repeatedly
        self.ui = sd.ui.WebGUI(self.config, self)

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
