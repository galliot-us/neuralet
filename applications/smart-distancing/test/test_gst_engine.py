import unittest
import doctest
import os
import sys

from typing import (
    Tuple,
    Optional,
    Iterator,
    Dict,
)

# base paths, and import setup
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

# where to find test data
TEST_DATA_DIR = os.path.join(THIS_DIR, 'data')
VIDEO_FILENAMES = (
    'TownCentreXVID.avi',
)


from smart_distancing.detectors.deepstream import _gst_engine

class TestGstEngine(unittest.TestCase):

    def test_doctests(self):
        """test none of the doctests fail"""
        self.assertEqual(
            doctest.testmod(_gst_engine)[0],
            0,
        )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
