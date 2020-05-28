"""DsDetector goes here"""

import logging
import itertools
import time

from smart_distancing.core import ConfigEngine
from smart_distancing.detectors import BaseDetector, OnFrameCallback
from smart_distancing.detectors.deepstream import DsEngine
from smart_distancing.detectors.deepstream import DsConfig

from smart_distancing.distance_pb2 import (
    Batch,
    Frame,
)

from typing import (
    Dict,
    Tuple,
    Sequence,
)

__all__ = ['DsDetector']

class DsDetector(BaseDetector):
    """
    DeepStream implementation of BaseDetector.
    """

    DEFAULT_MODEL_FILE = 'None'
    DEFAULT_MODEL_URL = 'None'

    engine = None  # type: DsEngine

    def __init__(self, config, on_frame:OnFrameCallback=None):
        # set the config
        self.config = config

        # assign the on_frame callback if any
        if on_frame:
            self.on_frame = on_frame
        else:
            self.on_frame = self._on_frame

        # set up a logger on the class
        self.logger = logging.getLogger(self.__class__.__name__)

        # add a frame counter
        self._frame_count = itertools.count()

    def load_model(self):
        """
        init/reinit a DsEngine instance (terminates if necessary).

        Called by start() automatically.
        """
        if self.engine and self.engine.is_alive():
            self.logger.info(
                "restarting engine")
            self.engine.terminate()
            self.engine.join()
        self.engine = DsEngine(DsConfig(self.config))

    # @Hossein I know the other classes don't have this, but it may make sense
    #  to add this start + stop functionality to the base class.
    def start(self, blocking=True, timeout=10):
        """
        Start DsDetector's engine.

        Arguments:
            blocking (bool):
                Whether to block while waiting for results. If false, busy waits
                with a sleep(0) in the loop (set True if you launch in a thread)
            timeout:
                If blocking is True,  
        """
        self.load_model()
        self.logger.info(
            f'starting up{" in blocking mode" if blocking else ""}')
        self.engine.blocking=blocking
        self.engine.start()
        self.engine.queue_timeout=10
        batch = Batch()
        while self.engine.is_alive():
            batch_str = self.engine.results
            if not batch_str:
                time.sleep(0)  # this is to switch context if launched in thread
                continue
            batch.ParseFromString(batch_str)
            for frame in batch.frames:  # pylint: disable=no-member
                next(self._frame_count)
                self.on_frame(frame)

    def stop(self):
        self.engine.stop()

    @property
    def fps(self):
        self.logger.warning("fps reporting not yet implemented")
        return 30

    @property
    def sources(self):
        self.logger.warning("getting sources at runtime not yet implemented")
        return []

    @sources.setter
    def sources(self, sources: Sequence[str]):
        self.logger.warning("setting sources at runtime not yet implemented")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
