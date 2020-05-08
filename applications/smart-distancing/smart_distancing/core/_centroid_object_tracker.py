# import the necessary packages
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist

__all__ = ['CentroidTracker']

class CentroidTracker:
    """
    A simple object tracker based on euclidean distance of bounding boxe centroids of two consecutive frames.
    if a box is lost between two frames the tracker keeps the box for next max_disappeared frames.

    :param max_disappeared: If a box is losted betweeb two frames the tracker keep the box for next
     max_disappeared frames.
    """

    def __init__(self, max_disappeared=50):
        self.nextobject_id = 0
        self.tracked_objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, object_item):
        # Register a new detected object and set a unique id for it
        self.tracked_objects[self.nextobject_id] = object_item
        self.disappeared[self.nextobject_id] = 0
        self.nextobject_id += 1

    def diregister(self, object_id):
        """
        Remove an object from objects and disappeared list.

        Args:
            object_id: Unique object ID for detected objects which is at objects list

        """
        del self.tracked_objects[object_id]
        del self.disappeared[object_id]

    def update(self, detected_objects):
        """
        Updates the objects from the previous frame.
        This function compares previous frame with current frame and take following actions:

        1- For each object that is being tracked, update its bounding box if it matches a
        detected object in the current frame.
        2- For any detection in the current frame that doesn't match any tracked object,
        register it as a new object.
        3- For each object that is being tracked and doesn't match a detection in the
        current frame, register it as lost and increment its disappeared counter.

        Args:
            detected_objects: List of detected objects.

        Return:
            tracked_objects: List of updated objects.
        """
        if len(detected_objects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Removed an object from the tracker when the object is missing in the 'max_disappeared' previous frames.
                if self.disappeared[object_id] > self.max_disappeared:
                    self.diregister(object_id)
            return self.tracked_objects

        input_centroids = np.zeros((len(detected_objects), 2))
        for i, object_item in enumerate(detected_objects):
            input_centroids[i] = (object_item["centroid"][0], object_item["centroid"][1])
        if len(self.tracked_objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(detected_objects[i])
        else:
            object_ids = list(self.tracked_objects.keys())
            object_centroids = [object_item["centroid"][0:2] for object_item in self.tracked_objects.values()]
            computed_dist = dist.cdist(np.array(object_centroids), input_centroids)
            rows = computed_dist.min(axis=1).argsort()
            cols = computed_dist.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.tracked_objects[object_id] = detected_objects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, computed_dist.shape[0])).difference(used_rows)
            unused_cols = set(range(0, computed_dist.shape[1])).difference(used_cols)

            if computed_dist.shape[0] >= computed_dist.shape[1]:
                biggest_existing_id = int(detected_objects[-1]["id"].split("-")[-1])

                for row in unused_rows:
                    object_id = object_ids[row]
                    self.tracked_objects[object_id]["id"] = self.tracked_objects[object_id]["id"].split("-")[
                                                                0] + "-" + str(
                        biggest_existing_id + 1)
                    self.disappeared[object_id] += 1
                    biggest_existing_id += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.diregister(object_id)

            else:
                for col in unused_cols:
                    self.register(detected_objects[col])

        return self.tracked_objects
