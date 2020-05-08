"""DeepStream Distancing implementation and related functions."""
import logging

import pyds

from smart_distancing.core._distancing import BaseDistancing

logger = logging.getLogger(__name__)

__all__ = ['DsDistancing']

class DsDistancing(BaseDistancing):
    """
    DeepStream implementation of Distancing.
    """

