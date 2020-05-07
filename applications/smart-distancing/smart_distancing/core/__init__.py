# common core imports
from smart_distancing.core._centroid_object_tracker import *
from smart_distancing.core._config_engine import *

# import Distancing base class
from smart_distancing.core._distancing import *

# import Distancing implementations
try:
    # OpenCV might not be installed
    from smart_distancing.core._cv_distancing import CvDistancing
except ImportError:
    CvDistancing = None
try:
    from smart_distancing.core._ds_distancing import DsDistancing
except ImportError:  # (DeepStream not found)
    DsDistancing = None
if CvDistancing is None and DsDistancing is None:
    raise ImportError("No available engines (OpenCV or DeepStream)")
# prefer DsDistancing if both are available (NVIDIA platforms)
if DsDistancing:
    DefaultDistancing = DsDistancing
else:
    DefaultDistancing = CvDistancing

__all__ = [
    'ConfigEngine',  # _config_engine.py
    'CentroidTracker',  # _centroid_object_tracker.py
    'BaseDistancing',  # _distancing.py
    'CvDistancing',  # _cv_distancing.py
    'DsDistancing',  # _ds_distancing.py
    'DefaultDistancing'  # this file
]
