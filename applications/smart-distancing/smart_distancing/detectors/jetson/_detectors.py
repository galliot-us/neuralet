import logging

from typing import List

import smart_distancing as sd

__all__ = [
    'JetsonDetector',
    'MobilenetSsdDetector',
]

logger = logging.getLogger(__name__)

class JetsonDetector(sd.detectors.BaseDetector):
    """Jetson sublcass of BaseDetector"""

    PLATFORM = 'jetson'
    # TODO(mdegans): secure hash verification of all models
    DEFAULT_MODEL_URL = 'https://github.com/Tony607/jetson_nano_trt_tf_ssd/raw/master/packages/jetpack4.3/'

    def load_model(self):
        """Implementation of load_model that starts a GStreamer subprocess"""
        pass

    @property
    def sources(self) -> List[str]:
        pass

    @sources.setter
    def sources(self, sources):
        pass

    @property
    def fps(self) -> int:
        pass


class MobilenetSsdDetector(JetsonDetector):
    """Mobilenet SSD implementation of JetsonDetector"""

    DEFAULT_MODEL_FILE = 'TRT_ssd_mobilenet_v2_coco.bin'
