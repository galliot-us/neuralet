"""Contains Detector, the base class for all detectors."""
import abc
import os
import logging
import urllib.parse
import urllib.request

import smart_distancing as sd



from typing import (
    List,
    Sequence,
    Tuple,
    Dict,
    Optional,
)

__all__ = ['BaseDetector', ]

logger = logging.getLogger(__name__)

class BaseDetector(abc.ABC):
    """
    A base class for all Detectors. The following should be overridden:

    PLATFORM the model platform (eg. edgetpu, jetson, x86)
    DEFAULT_MODEL_FILE with the desired model basename
    DEFAULT_MODEL_URL with the url path minus filename of the model

    load_model() to load the model. This is called for you on __init__.

    Something should also call on_frame() with a sequence of sd.Detection
    """

    PLATFORM = None  # type: Tuple
    DEFAULT_MODEL_FILE = None  # type: str
    DEFAULT_MODEL_URL = None  # type: str

    def __init__(self, config):
        # set the config
        self.config = config

        # download the model if necessary
        if not os.path.isfile(self.model_file):
            logger.info(
                f'model does not exist under: "{self.model_path}" '
                f'downloading from  "{self.DEFAULT_MODEL_URL}"')
            os.makedirs(self.model_path, mode=0o755, exist_ok=True)
            urllib.request.urlretrieve(self.model_url, self.model_file)

        # load the model
        self.load_model()

    @property
    def detector_config(self) -> Dict:
        """:return: the 'Detector' section from self.config"""
        return self.config.get_section_dict('Detector')

    @property
    def name(self) -> str:
        """:return: the detector name."""
        return self.detector_config['Name']

    @property
    def model_path(self) -> Optional[str]:
        """:return: the folder containing the model."""
        try:
            cfg_model_path = self.detector_config['ModelPath']
            if cfg_model_path:  # not None and greater than zero in length
                return cfg_model_path
        except KeyError:
            pass
        return os.path.join(sd.MODEL_DIR, self.PLATFORM)

    @property
    def model_file(self) -> Optional[str]:
        """:return: the model filename."""
        return os.path.join(self.model_path, self.DEFAULT_MODEL_FILE)

    @property
    def model_url(self) -> str:
        """:return: a parsed url pointing to a downloadable model"""
        # this is done to validate it's at least a valid uri
        # TODO(mdegans?): move to config class
        return urllib.parse.urlunparse(urllib.parse.urlparse(
            self.DEFAULT_MODEL_URL + self.DEFAULT_MODEL_FILE))

    @property
    def class_id(self) -> int:
        """:return: the class id to detect."""
        return int(self.detector_config['ClassID'])

    @property
    def score_threshold(self) -> float:
        """:return: the detection minimum threshold (MinScore)."""
        return float(self.detector_config['MinScore'])
    min_score = score_threshold   # an alias for ease of access

    @property
    @abc.abstractmethod
    def sources(self) -> List[str]:
        """:return: the active sources."""

    @sources.setter
    @abc.abstractmethod
    def sources(self, source: Sequence[str]):
        """Set the active sources"""

    @property
    @abc.abstractmethod
    def fps(self) -> int:
        """:return: the current fps"""

    @abc.abstractmethod
    def load_model(self):
        """load the model. Called by the default implementation of __init__."""

    def on_frame(self, detections: sd.Detections):
        """
        Calculate distances between detections and updates UI. Default concrete
        implementation. Should be called by the subclass on every frame with
        an iterable of Detection.

        :param detections: a Sequence of sd.Detection
        """
