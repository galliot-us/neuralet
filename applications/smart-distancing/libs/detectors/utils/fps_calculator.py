"""A set of function(s) that are used for estimating frame per second (fps).

These function(s) often receive an inference time, perform some calculation on it.
The function(s) do return a fps value.

"""


def _convert_infr_time_to_fps(infr_time: float) -> float:
    # Gets the time of inference (infr_time) and returns Frames Per Second (fps)
    fps = 1.0 / infr_time
    return fps
