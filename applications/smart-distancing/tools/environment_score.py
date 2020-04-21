"""
Documentation after implementing
"""
import numpy as np

MAX_ACCEPTABLE_CAPACITY = 10  # The maximum number of people that can stand as far away from other people as possible
MAX_CAPACITY = 20  # The maximum number of people in the environment


def mx_environment_scoring_consider_crowd(detected_pedestrians: int, violated_pedestrians: int) -> np.float64:
    env_score = 1 - np.minimum(
        ((violated_pedestrians + detected_pedestrians) / (MAX_CAPACITY + MAX_ACCEPTABLE_CAPACITY)), 1)
    return env_score


def mx_environment_scoring(violated_pedestrians: int) -> np.float64:
    env_score = 1 - np.minimum((violated_pedestrians / MAX_CAPACITY), 1)
    return env_score
