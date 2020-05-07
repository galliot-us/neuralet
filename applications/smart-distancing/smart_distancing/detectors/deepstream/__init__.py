# GStreamer needs to be imported before pyds or else there is crash on Gst.init
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import (
    Gst,
    GLib,
)
from smart_distancing.detectors.deepstream._pyds import *
from smart_distancing.detectors.deepstream._ds_config import *
from smart_distancing.detectors.deepstream._ds_engine import *

__all__ = [
    'DsConfig',  # _ds_config.py
    'ElemConfig',  # _ds_config.py
    'GstConfig',  # _ds_config.py
    'pyds',  # _pyds.py
    'PYDS_PATH',  # _pyds.py
    'PYDS_INSTRUCTIONS',  # _pyds.py
    'GstEngine',  # _ds_engine.py
    'link_many',  # _ds_engine.py
    'frame_meta_iterator',  # _ds_engine.py
    'obj_meta_iterator',  # _ds_engine.py
]
