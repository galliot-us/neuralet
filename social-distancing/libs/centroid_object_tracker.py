# import the necessary packages
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    """
    a simple object tracker based on Euclidian distance of bounding boxes centroid of two consecutive frames.
    if a box is losted betweeb two frames the tracker keep the box for next maxDisappeared frames.
    """

    def __init__(self, maxDisappeared=50):

        """
        maxDisappeared:if a box is losted betweeb two frames the tracker keep the box for next maxDisappeared frames.

        """
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, object_item):
        self.objects[self.nextObjectID] = object_item
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, object_list):
        if len(object_list) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(object_list), 2))
        for i, object_item in enumerate(object_list):
            inputCentroids[i] = (object_item["centroid"][0], object_item["centroid"][1])
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(object_list[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [object_item["centroid"][0:2] for object_item in self.objects.values()]
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = object_list[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                biggest_existing_id = int(object_list[-1]["id"].split("-")[-1])

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.objects[objectID]["id"] = self.objects[objectID]["id"].split("-")[0] + "-" + str(
                        biggest_existing_id + 1)
                    self.disappeared[objectID] += 1
                    biggest_existing_id += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(object_list[col])

        return self.objects
