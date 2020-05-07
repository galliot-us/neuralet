"""
Python DeepStream bindings loader.

This is necessary because Nvidia has no proper python package for the bindings
and a straight-up "import pyds" will not work without sys.path hackery.

Attributes:
    PYDS_PATH (str):
        The platform specific path to pyds.so (the bindings).
        This path is inserted into sys.path.
    PYDS_INSTRUCTIONS (str):
        URI for (overly complicated) Installation instructions
        for the Python DeepStream bindings.
"""
import os
import logging
import sys
import platform

logger = logging.getLogger()

__all__ = [
    'PYDS_PATH',
    'PYDS_INSTRUCTIONS',
    'pyds'
]

# Python DeepStream paths
PYDS_ROOT = '/opt/nvidia/deepstream/deepstream/sources/python/bindings'
PYDS_JETSON_PATH = os.path.join(PYDS_ROOT, 'jetson')
PYDS_x86_64_PATH = os.path.join(PYDS_ROOT, 'x86_64')
# Installing the bindings is actually fairly cumbersome, unfortunately. Please, Nvidia
# put more effort into testing and packaging your products. Speed is not enough.
PYDS_INSTRUCTIONS = 'https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md#running-sample-applications'
if platform.machine() == 'aarch64':
    PYDS_PATH = PYDS_JETSON_PATH
elif platform.machine() == 'x86_64':
    PYDS_PATH = PYDS_x86_64_PATH
else:
    logger.warning(
        f"unsupported platform for DeepStream Python bindings")
    PYDS_PATH = None

if PYDS_PATH:
    sys.path.insert(0, PYDS_PATH)
    try:
        import pyds
    except ImportError:
        logger.warning(
            f'pyds could not be imported. '
            f'install instructions: {PYDS_INSTRUCTIONS}')
        raise
