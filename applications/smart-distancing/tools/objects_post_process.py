"""
This module includes set of functions that apply as a post-processing to the detectors output
"""
import numpy as np


def extract_violating_objects(distances, dist_threshold):
    """Extract pair of objects that are closer than the distance threshold.

    Args:
        distances: A 2-d numpy array that stores distance between each pair of objects.
        dist_threshold: the minimum distance for considering unsafe distance between objects

    Returns:
        violating_objects: A 2-d numpy array where each row is the ids of the objects that violated the social distancing.

    """
    triu_distances = np.triu(distances) + np.tril(10 * np.ones(distances.shape))
    violating_objects = np.argwhere(triu_distances < float(dist_threshold))
    return violating_objects
