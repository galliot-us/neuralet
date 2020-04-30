"""Utilities to serialize data for the log."""
from smart_distancing import Detections
from typing import (
    Iterator,
)

import json

__all__ = [
    'serialize',
    'serialize_iter',
]

# alias for json.dumps
serialize = json.dumps

def serialize_iter(detections: Detections) -> Iterator[str]:
    """
    An iterator that serializes detection data as json.

    :yield: a json string for each detection in the supplied iterable.
    """
    for d in detections:
        yield serialize(d)
