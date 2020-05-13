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


class TestDsEngineInteractive(unittest.TestCase):
    """
    these tests should be run and killed interactively
    """

    def test_results_queue(self):
        """test runner should ensure results are printed out when the pipeline starts"""
        # TODO(mdegans): automation
        master_config = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = DsConfig(master_config)
        engine = DsEngine(config)
        engine.queue_timeout = 1000
        engine.start()
        try:
            while True:
                time.sleep(1)
                results = engine.results
                if results:
                    print(results)
        except KeyboardInterrupt:
            engine.stop()
            engine.join()


def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestDsEngineFast())
    return suite

def medium_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestDsEngineSlow())
    return suite

def interactive_suite():
    suite = unittest.TestSuite()
    suite.addTest(TestDsEngineInteractive())
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
        runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestDsEngineInteractive))
    else:
        runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestDsEngineFast))
        runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(TestDsEngineSlow))
