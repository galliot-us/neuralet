"""
Top level for smart_distancing.

Contains typing types and some globals.

Attributes:
    PACKAGE_ROOT (str):
        Absolute path to the package root.
    PACKAGE_DATA_ROOT (str):
        Absolute path to package data dir.
    PACKAGE_MODEL_ROOT (str):
        Absolute path to model dir.
    USER_ROOT (str):
        Absolute path to user root (~/.smart_distancing).
    USER_CONFIG_DIR (str):
        Absolute path to user config dir (under USER_ROOT).
    USER_MODEL_DIR (str):
        Absolute path to user model dir (under USER_ROOT).
    Detection (:obj:`typing.Type`):
        Typing module Type of a detection (a dict of str:Any).
    Detections (:obj:`typing.Type`):
        Typing module type of a Sequence (eg. tuple, list) of Detection.
"""
import os

from typing import (
    Any,
    Sequence,
    MutableMapping,
)
# custom typing
Detection = MutableMapping[str, Any]
Detections = Sequence[Detection]

import smart_distancing.detectors
import smart_distancing.loggers
import smart_distancing.ui
import smart_distancing.core


# module/system level variables
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
        __version__ = version_file.readline().strip()[:16]
except OSError:
    __version__ = 'fallback (VERSION file missing from package root?)'
__all__ = [
    'Detection',
    'Detections',
    'PACKAGE_ROOT',
    'USER_ROOT',
    'USER_CONFIG_DIR',
    'USER_MODEL_DIR',
]

# module/system level variables
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DATA_ROOT = os.path.join(PACKAGE_ROOT, 'data')
PACKAGE_MODEL_ROOT = os.path.join(PACKAGE_DATA_ROOT, 'models')

USER_ROOT = os.path.join(os.path.expanduser('~'), '.smart_distancing')
USER_CONFIG_DIR = os.path.join(USER_ROOT, 'configs')
USER_MODEL_DIR = os.path.join(USER_ROOT, 'models')
