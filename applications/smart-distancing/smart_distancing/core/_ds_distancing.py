"""DeepStream Distancing implementation and related functions."""
import logging
import queue

from smart_distancing.detectors.deepstream import (
    pyds,
)

from smart_distancing.core._distancing import BaseDistancing
import smart_distancing.detectors.deepstream as ds

logger = logging.getLogger(__name__)

__all__ = ['DsDistancing']

class DsDistancing(BaseDistancing):
    """
    DeepStream implementation of Distancing.
    """

    ds_config = None  #type: ds.DsConfig

    def process_video(self, video_path:str):
        self.detector = ds.DsDetector(self.config)

        # while the engine is alive
        while self.detector.engine.is_alive() and self.running_video:
            try:
                detections = self.detector.engine.results
            except TimeoutError:
                logger.warning(f'engine timed out')
                self.detector.restart()
                # wait for engine 
            if detections:
                print(detections)
                # self.ui.update(detections)
