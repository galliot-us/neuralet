"""
The DeepStream detector module includes a DeepStream specific implementation
of the BaseDetector class and various utility classes and functions.
"""
# GStreamer needs to be imported before pyds or else there is crash on Gst.init
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import (
    Gst,
    GLib,
)
from smart_distancing.detectors.deepstream._ds_utils import *
from smart_distancing.detectors.deepstream._pyds import *
from smart_distancing.detectors.deepstream._ds_config import *
from smart_distancing.detectors.deepstream._ds_engine import *
from smart_distancing.detectors.deepstream._ds_engine import *
from smart_distancing.detectors.deepstream._detectors import *

__all__ = [
    'DsConfig',  # _ds_config.py
    'find_deepstream',  # _ds_utils.py
    'DsDetector',  # _detectors.py
    'DsEngine',  # _ds_engine.py
    'ElemConfig',  # _ds_config.py
    'frame_meta_iterator',  # _ds_engine.py
    'GstConfig',  # _ds_config.py
    'GstEngine',  # _ds_engine.py
    'link_many',  # _ds_engine.py
    'obj_meta_iterator',  # _ds_engine.py
    'PYDS_INSTRUCTIONS',  # _pyds.py
    'PYDS_PATH',  # _pyds.py
    'pyds',  # _pyds.py
]
