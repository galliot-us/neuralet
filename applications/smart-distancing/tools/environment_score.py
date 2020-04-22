"""
Set of functions related to calculating environment score for physical distancing is implemented here.
"""
import numpy as np

# TODO: consider these as a parameter; In the future, depend on the project, they may be moved at a separate config file
MAX_ACCEPTABLE_CAPACITY = 20  # The maximum number of people that can stand as far away from other people as possible
MAX_CAPACITY = 60  # The maximum number of people in the environment


def mx_environment_scoring_consider_crowd(
    detected_pedestrians: int, violated_pedestrians: int
) -> np.float64:
    """
    This function calculates the environment score based on the crowd and acceptable number of objects, and violating
    objects. The maximum capacity is considered for determining how the environment situation is critical.

    Args:
        detected_pedestrians: Number of detected objects (people)
        violated_pedestrians: Number of violating objects (people)

    Returns:
        env_score: The normalized score of environment (0 is bad and 1 is good)

    """
    env_score = 1 - np.minimum(
        (
            (violated_pedestrians + detected_pedestrians)
            / (MAX_CAPACITY + MAX_ACCEPTABLE_CAPACITY)
        ),
        1,
    )
    env_score = np.round(env_score, 2)
    return env_score


def mx_environment_scoring(violated_pedestrians: int) -> np.float64:
    """
    This function calculates the environment score based on acceptable number of object in an environment,
    and the violating objects.

    Args:
        violated_pedestrians: Number of violating objects (people)

    Returns:
        env_score: The normalized score of environment (0 is bad and 1 is good)

    """
    env_score = 1 - np.minimum((violated_pedestrians / MAX_ACCEPTABLE_CAPACITY), 1)
    env_score = np.round(env_score, 2)
    return env_score
