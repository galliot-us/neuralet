"""DeepStream Distancing implementation and related functions."""
import logging
import queue

from smart_distancing.detectors.deepstream import (
    pyds,
)

from smart_distancing.utils.visualization_utils as vis_util
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

        while self.detector.engine.is_alive() and self.running_video:
            detections = self.engine.results
            if detections:
                self._on_detections(detections)
                # we spin here on purpose

    def _on_detections(self, detections):
        w, h = self.resolution

        for uid, obj in detections.items():
            box = obj["bbox"]
            x0 = box[1]
            y0 = box[0]
            x1 = box[3]
            y1 = box[2]
            obj["centroid"] = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
            obj["bbox"] = [x0, y0, x1, y1]
            obj["centroidReal"] = [(x0 + x1) * w / 2, (y0 + y1) * h / 2, (x1 - x0) * w, (y1 - y0) * h]
            obj["bboxReal"] = [x0 * w, y0 * h, x1 * w, y1 * h]

        # calculate pair distancings
        objects = self.calculate_distancing(detections)
        # score objects
        scored_objects = vis_util.visualization_preparation(objects, distances, self.ui._dist_threshold)
        # a record to log and send back to the process
        record = {
            'fnum': detections['fnum'],
            'scored_objects': scored_objects,
        }
        logger.debug(record)
        try:
            # send the results back to the GstEngine (non-blocking)
            self.detector.engine.osd_queue.put_nowait(record)
        except queue.Full:
            # this happens when we calculate too quickly for the GstEngine
            pass
        # update the ui no matter what
        self.ui.update(None, scored_objects)
