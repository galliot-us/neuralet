import unittest
import urllib.request
import time
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

SKELETON_CONFIG = os.path.join(
    PARENT_DIR, 'smart_distancing', 'data', 'config', 'skeleton.in')
DS_ONE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_one_source.ini')

# where to find test data
TEST_DATA_DIR = os.path.join(THIS_DIR, 'data')
# put video filesname here.when added to 'data':
VIDEO_FILENAMES = (
    'TownCentreXVID.avi',
)
VIDEO_FILENAMES = tuple(
    os.path.join(TEST_DATA_DIR, fn) for fn in VIDEO_FILENAMES)
NETWORK_TEST_HOST = 'http://www.freedesktop.org/'
TEST_VID_URI = 'https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm'

from smart_distancing.core import ConfigEngine
from smart_distancing.detectors.deepstream import _gst_engine
from smart_distancing.detectors.deepstream._gst_engine import GstEngine
from smart_distancing.detectors.deepstream._ds_config import GstConfig

def freedesktop_is_up():
    """
    Test if our freedesktop.org is online and accessable,
    because it frequently is not. Test shouldn't fail because of
    potato servers.
    """
    try:
        urllib.request.urlopen(NETWORK_TEST_HOST, timeout=2)
        return True
    except urllib.request.URLError: 
        return False

class TestGstEngine(unittest.TestCase):

    def setUp(self):
        self.conf_one_source = ConfigEngine(DS_ONE_SOURCE_CONFIG)

    def test_doctests(self):
        """test none of the doctests fail"""
        #FIXME(mdegans): documentation needs to be updated with an examples
        self.assertEqual(
            doctest.testmod(_gst_engine)[0],
            0,
        )

    def test_local_single(self):
        config = GstConfig(self.conf_one_source)
        engine = GstEngine(config)
        engine.start()
        time.sleep(1)
        engine.stop()
        engine.join(10)
        self.assertEqual(engine.exitcode, 0)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
