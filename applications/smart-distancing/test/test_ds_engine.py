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
CONFIG_DIR = os.path.join(PARENT_DIR, 'smart_distancing', 'data', 'config')

import smart_distancing as sd
from smart_distancing.detectors.deepstream import _ds_engine
from smart_distancing.detectors.deepstream import DsEngine, DsConfig

DS_ONE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_one_source.ini')
DS_TWO_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_two_source.ini')
DS_THREE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_three_source.ini')

TEST_DS_VIDEO = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

class TestDsEngineFast(unittest.TestCase):

    def test_doctests(self):
        """test none of the doctests fail"""
        # FIXME(mdegans):
        #  * add doctests
        #  * check returncode works as expected
        #    (modify bus callback so the process returns error on fail)
        self.assertEqual(
            doctest.testmod(_ds_engine, optionflags=doctest.ELLIPSIS)[0],
            0,
        )

class TestDsEngineSlow(unittest.TestCase):

    def setUp(self):
        self.confs = {
            1: sd.core.ConfigEngine(DS_ONE_SOURCE_CONFIG),
            2: sd.core.ConfigEngine(DS_TWO_SOURCE_CONFIG),
            3: sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG),
        }


    def test_start_stop(self):
        """
        this test might take a long time
        """
        for num, master_config in self.confs.items():
            with self.subTest(num):
                config = DsConfig(master_config)
                engine = DsEngine(config)
                engine.start()
                engine.stop()
                engine.join()
                self.assertEqual(0, engine.exitcode)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--slow', action='store_true')
    # args = ap.parse_args()

    # tests = []

    # fast = unittest.TestSuite()
    # fast.addTest(TestDsEngineFast())
    # tests.append(fast)

    # if args.slow:
    #     slow = unittest.TestSuite()
    #     slow.addTest(TestDsEngineSlow())
    #     tests.append(slow)

    # for suite in tests:
    #     unittest.TextTestRunner().run(suite)

    unittest.main()
