"""
DeepStream common utilities.
"""
import os

from typing import (
    Tuple,
)

__all__ = ['find_deepstream']

DS_VERSIONS = ('4.0', '5.0')
DS4_PATH = '/opt/nvidia/deepstream/deepstream-{ver}'

def find_deepstream() -> Tuple[str, str]:
    """
    Finds DeepStream.

    Return:
        A 2 tuple of the DeepStream version
        and it's root path or None if no
        version is found.
    """
    # TODO(mdegans): implement
    for ver in DS_VERSIONS:
        ds_dir = DS4_PATH.format(ver=ver)
        if os.path.isdir(ds_dir):
            return ver, ds_dir
