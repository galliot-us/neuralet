"""Utilities to serialize data for the log."""
import logging

from smart_distancing import Detections
from typing import (
    Iterator,
)

import json

__all__ = [
    'serialize',
    'serialize_iter',
    'spam_filter',
]

# alias for json.dumps
serialize = json.dumps

def serialize_iter(detections: Detections) -> Iterator[str]:
    """
    An iterator that serializes detection data as json.

    :yield: a json string for each detection in the supplied iterable.

    >>> l = ['spam', 'ham']
    >>> d = {'foo': 'bar'}
    >>> rec = [l, d]
    >>> list(serialize_iter(rec))
    ['["spam", "ham"]', '{"foo": "bar"}']
    """
    for d in detections:
        yield serialize(d)

def spam_filter(record: logging.LogRecord) -> bool:
    """
    A logging filter function to filter out log spam.

    Return True if a record should be kept, False if it should be discarded.

    Ignores if:
         record.name == 'PngImagePlugin'
    """
    if record.module == 'PngImagePlugin':
        return False
    return True

if __name__ == "__main__":
    import doctest
    doctest.testmod()
