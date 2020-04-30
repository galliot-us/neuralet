"""DeepStream Distancing implementation and related functions."""
import logging

# import pyds

import smart_distancing as sd

logger = logging.getLogger(__name__)

__all__ = ['DsDistancing']

class DsDistancing(sd.core.BaseDistancing):
    """
    DeepStream implementation of Distancing.
    """

