"""
Top level for smart_distancing.

Contains typing types and some globals.

Attributes:

    PACKAGE_ROOT (str):
        Absolute path to the package root.

    PACKAGE_DATA_ROOT (str):
        Absolute path to package data dir.
    PACKAGE_CONFIG_DIR (str):
        Absolute path to package config dir (data subdir).
    PACKAGE_MODEL_ROOT (str):
        Absolute path to package model dir (data subdir).
    PACKAGE_LOG_DIR (str):
        Absolute path to package log dir (data subdir).

    USER_ROOT (str):
        Absolute path to user root (~/.smart_distancing).
    USER_CONFIG_DIR (str):
        Absolute path to user config dir (under USER_ROOT).
    USER_MODEL_DIR (str):
        Absolute path to user model dir (under USER_ROOT).
    USER_LOG_DIR (str):
        Absolute path to user log dir (under USER_ROOT).

    DEFAULT_TO_USER_PATHS (bool):
        If true, the default paths (below) are user paths.
        This may be toggled if any path is not writable.
    DATA_ROOT (str):
        Default data root.
    CONFIG_DIR (str):
        Default config dir.
    MODEL_DIR (str):
        Default model dir.
    LOG_DIR (str):
        Default log dir.

    Detection (:obj:`typing.Type`):
        Typing module Type of a detection (a dict of str:Any).
    Detections (:obj:`typing.Type`):
        Typing module type of a Sequence (eg. tuple, list) of Detection.
"""
import os
import sys

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

# Default paths to try:
DEFAULT_TO_USER_PATHS = True

# Where data can go if the path is writable
PACKAGE_DATA_ROOT = os.path.join(PACKAGE_ROOT, 'data')
PACKAGE_CONFIG_DIR = os.path.join(PACKAGE_DATA_ROOT, 'config')
PACKAGE_MODEL_DIR = os.path.join(PACKAGE_DATA_ROOT, 'models')
PACKAGE_LOG_DIR = os.path.join(PACKAGE_DATA_ROOT, 'logs')

PACKAGE_PATHS = (
    PACKAGE_DATA_ROOT, PACKAGE_CONFIG_DIR, PACKAGE_MODEL_DIR, PACKAGE_LOG_DIR,
)

# Where data can go if the path is not writable
USER_ROOT = os.path.join(os.path.expanduser('~'), '.smart_distancing')
USER_CONFIG_DIR = os.path.join(USER_ROOT, 'configs')
USER_MODEL_DIR = os.path.join(USER_ROOT, 'models')
USER_LOG_DIR = os.path.join(USER_ROOT, 'logs')

USER_PATHS = (
    USER_ROOT, USER_CONFIG_DIR, USER_MODEL_DIR, USER_LOG_DIR,
)

def _make_paths(paths: Sequence[str]):
    """make some paths, return false on failure"""
    try:
        for path in paths:
            os.makedirs(path, mode=0o755, exist_ok=True)
            if not os.access(path, os.W_OK):
                return False
    except OSError:
        return False
    return True

# Try to make the default paths. If it fails, use the others as fallback.
if not _make_paths(USER_PATHS if DEFAULT_TO_USER_PATHS else PACKAGE_PATHS):
    DEFAULT_TO_USER_PATHS = not DEFAULT_TO_USER_PATHS

# set the global default paths acordingly
DATA_ROOT = USER_ROOT if DEFAULT_TO_USER_PATHS else PACKAGE_DATA_ROOT
CONFIG_DIR = USER_CONFIG_DIR if DEFAULT_TO_USER_PATHS else PACKAGE_CONFIG_DIR
MODEL_DIR = USER_MODEL_DIR if DEFAULT_TO_USER_PATHS else PACKAGE_MODEL_DIR
LOG_DIR = USER_LOG_DIR if DEFAULT_TO_USER_PATHS else PACKAGE_LOG_DIR
