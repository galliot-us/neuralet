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

__version__ = '0.1.0'
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
