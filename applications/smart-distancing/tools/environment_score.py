"""
Set of functions related to calculating environment score for physical distancing is implemented here.
"""
import numpy as np

# TODO: consider these as a parameter; In the future, depend on the project, they may be moved at a separate config file
MAX_ACCEPTABLE_CAPACITY = 20  # The maximum number of people that can stand as far away from other people as possible
MAX_CAPACITY = 60  # The maximum number of people in the environment


def mx_environment_scoring_consider_crowd(
        detected_pedestrians: int, violating_pedestrians: int
) -> np.float64:
    """
    This function calculates the environment score based on the crowd and acceptable number of objects, and violating
    objects. The maximum capacity is considered for determining how the environment situation is critical.

    Args:
        detected_pedestrians: Number of detected objects (people)
        violating_pedestrians: Number of violating objects (people)

    Returns:
        env_score: The normalized score of environment (0 is bad and 1 is good)

    Examples:
        >>> print(mx_environment_scoring_consider_crowd(detected_pedestrians=10, violating_pedestrians=5))
        0.81  # The score is 0.81 because there are 10 people in the environment and 5 people don't keep
         physical distances
        >>> print(mx_environment_scoring_consider_crowd(detected_pedestrians=5, violating_pedestrians=5))
        0.88  # The number of violating pedestrians is as the same previous example, however because of decreasing
         the number of people in the environment the score is increased
        >>> print(mx_environment_scoring_consider_crowd(detected_pedestrians=20, violating_pedestrians=5))
        0.69  # In this example the people is increased and the environment is more risky, as you see the
        score is decreased
        >>> print(mx_environment_scoring_consider_crowd(detected_pedestrians=20, violating_pedestrians=20))
        0.5  #  This is an example of non-crowded environment in which all people does not keep physical distances
        >>> print(mx_environment_scoring_consider_crowd(detected_pedestrians=60, violating_pedestrians=20))
        0.0  # This is an example of rush hours in which the number of people is equal to MAX_CAPACITY and
         also the number of violating people is equal to MAX_ACCEPTABLE_CAPACITY. Score < 0.5 means that the
         environment is a crowded environment


    """
    env_score = 1 - np.minimum(
        (
                (violating_pedestrians + detected_pedestrians)
                / (MAX_CAPACITY + MAX_ACCEPTABLE_CAPACITY)
        ),
        1,
    )
    env_score = np.round(env_score, 2)
    return env_score


def mx_environment_scoring(violating_pedestrians: int) -> np.float64:
    """
    This function calculates the environment score based on acceptable number of object in an environment,
    and the violating objects.
    This function returns 0 if the number of violating people will be equal to MAX_ACCEPTABLE_CAPACITY

    Args:
        violating_pedestrians: Number of violating objects (people)

    Returns:
        env_score: The normalized score of environment (0 is bad and 1 is good)

    """
    env_score = 1 - np.minimum((violating_pedestrians / MAX_ACCEPTABLE_CAPACITY), 1)
    env_score = np.round(env_score, 2)
    return env_score
