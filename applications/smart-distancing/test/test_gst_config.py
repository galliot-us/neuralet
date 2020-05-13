import unittest
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

CONFIG_FILES = []
for root, _, fns in os.walk(CONFIG_DIR):
    for fn in fns:
        CONFIG_FILES.append(os.path.join(CONFIG_DIR, fn))

SKELETON_CONFIG = os.path.join(
    PARENT_DIR, 'smart_distancing', 'data', 'config', 'skeleton.in')

DS_ONE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_one_source.ini')
DS_TWO_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_two_source.ini')
DS_THREE_SOURCE_CONFIG = os.path.join(
    THIS_DIR, 'data', 'deepstream_three_source.ini')

TEST_DS_VIDEO = '/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.h264'

import smart_distancing as sd

class TestGstConfig(unittest.TestCase):
    # NOTE(mdegans) this could probably use some common setup and teardown

    def test_sources_required(self):
        master_cfg = sd.core.ConfigEngine(SKELETON_CONFIG)
        with self.assertRaises(ValueError) as cm:
            config = sd.detectors.deepstream.GstConfig(master_cfg)
        self.assertIn('at least one', cm.exception.args[0])

    def test_sources_property(self):
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.GstConfig(master_cfg)
        with self.subTest('number'):
            # test multiple sections work
            self.assertEqual(len(config.src_configs), 3)
        with self.subTest('correct value'):
            # test the uri read is correct
            for conf in config.src_configs:
                self.assertEqual(
                    conf['uri'],
                    f'file://{TEST_DS_VIDEO}',)

    def test_out_resolution_property(self):
        master_cfg = sd.core.ConfigEngine(DS_ONE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.GstConfig(master_cfg)
        self.assertEqual(
            config.out_resolution,
            (640, 480),
        )

    def test_rows_and_columns_property(self):
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.GstConfig(master_cfg)
        self.assertEqual(
            config.rows_and_columns,
            2,
        )

    def test_tile_resolution_property(self):
        # TODO(mdegans): write better test
        # this also tests out_resolution
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.GstConfig(master_cfg)
        self.assertEqual(
            config.tile_resolution,
            (320, 240),
        )

    def test_blank_properties(self):
        # these should all return an empty dict
        blank_props = (
            'muxer_config',
            'tracker_config',
            'osd_config',
            'sink_config'
        )
        master_cfg = sd.core.ConfigEngine(DS_ONE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.GstConfig(master_cfg)
        for name in blank_props:
            with self.subTest(name):
                actual = getattr(config, name)
                self.assertEqual(
                    actual,
                    dict(),
                )

class TestDsConfig(unittest.TestCase):

    def test_batch_size_property(self):
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.DsConfig(master_cfg)
        self.assertEqual(
            config.batch_size,
            4,
        )

    def test_muxer_configs_property(self):
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.DsConfig(master_cfg)
        expected = {
            'width': 320,
            'height': 240,
            'batch-size': 4,
        }
        self.assertEqual(
            config.muxer_config,
            expected,
        )

    def test_infer_configs_property(self):
        master_cfg = sd.core.ConfigEngine(DS_THREE_SOURCE_CONFIG)
        config = sd.detectors.deepstream.DsConfig(master_cfg)
        expected = {
            'config-file-path': config.RESNET_CONF,
            'batch-size': 4,
        }
        self.assertEqual(
            config.infer_configs[0],
            expected,
        )

    

if __name__ == "__main__":
    unittest.main(verbosity=3)
