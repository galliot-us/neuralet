import unittest
import doctest
import os
import sys
import time

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
CONFIG_DIR = os.path.join(PARENT_DIR, 'smart_distancing', 'data', 'config')

import smart_distancing as sd

from smart_distancing.detectors.deepstream import _ds_engine
from smart_distancing.detectors.deepstream import DsDetector
from smart_distancing.distance_pb2 import Frame


DS_ONE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_one_source.ini')
DS_TWO_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_two_source.ini')
DS_THREE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_three_source.ini')

TEST_DS_VIDEO = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

class TestDsDetectorInteractive(unittest.TestCase):
    """
    these tests should be run and killed interactively
    """

    def test_results_queue(self):
        """test runner should ensure results are printed out when the pipeline starts"""
        time.sleep(3)  # to give time to read the above message
        # TODO(mdegans): automation
        config = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        detector = DsDetector(config)
        detector.start()

    def test_on_frame(self):
        """test runner should ensure results are pretty printed"""
        time.sleep(3)  # to give time to read the above message
        def on_frame(frame: Frame):
            print(frame)
        config = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        detector = DsDetector(config, on_frame=on_frame)
        detector.start()

def interactive_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestDsDetectorInteractive())
    return suite

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--interactive', action='store_true')
    args = ap.parse_args()

    runner = unittest.TextTestRunner()

    if args.interactive:
        runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestDsDetectorInteractive))
    else:
        import sys
        sys.stderr.write("TODO(mdegans): implement fast tests (try with --interactive for slow ones)\n")
