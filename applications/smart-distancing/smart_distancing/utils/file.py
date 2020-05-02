"""This file contains some file/path utilities"""
import os

import smart_distancing as sd

from typing import (
    Iterator,
    List,
)


def get_ini_files(dir_) -> Iterator[str]:
    """
    Iterate through .ini files found in a path.
    Does not recurse through subdirs.

    >>> list(get_ini_files(sd.PACKAGE_CONFIG_DIR))
    ['skeleton.ini', 'x86.ini', 'jetson.ini']
    """
    _, _, filenames = next(os.walk(dir_))
    for filename in filenames:
        if filename.endswith('.ini'):
            yield filename


def config_files(include_fallback=True) -> List[str]:
    """
    :return: a list of all available .ini config files.

    :param include_fallback: if True, includes sd.FALLBACK_CONFIG_DIR

    >>> config_files()
    ['jetson.ini', 'skeleton.ini', 'x86.ini']
    """
    inis = set(get_ini_files(sd.CONFIG_DIR))
    if include_fallback:
        fallback = set(get_ini_files(sd.FALLBACK_CONFIG_DIR))
        inis |= fallback
    return sorted(inis)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
