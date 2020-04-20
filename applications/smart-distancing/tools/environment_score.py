"""
Documentation after implementing
"""
import numpy as np

MAX_CAPACITY = 10  # The maximum number of people that can stand as far away from other people as possible


def max_capasity_environment_scoring(violating_pedestrians: int, ) -> np.float:
    env_score = 1 - np.minimum((MAX_CAPACITY / violating_pedestrians), 1)
    return env_score
