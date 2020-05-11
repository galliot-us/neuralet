"""DsDetector goes here"""

from smart_distancing.core import ConfigEngine
from smart_distancing.detectors import BaseDetector
from smart_distancing.detectors.deepstream import DsEngine
from smart_distancing.detectors.deepstream import DsConfig

from typing import (
    Dict,
    Tuple,
)

__all__ = ['DsDetector']

class DsDetector(BaseDetector):
    """
    DeepStream implementation of BaseDetector.
    """

    engine = None  # type: DsEngine

    def load_model(self):
        ds_config = (self.config)
        self.engine = DsEngine(ds_config)
        self.engine.start()

    def inference(self):
        self.on_frame(self.engine.results)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
