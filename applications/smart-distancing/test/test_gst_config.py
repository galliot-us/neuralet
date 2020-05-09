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

import smart_distancing as sd

class TestGstConfig(unittest.TestCase):
    """
    This tests GstConfig.
    
    It makes some assumptions about defaults that are fragile, but
    if any .ini configs have non-standard defaults, it'll also alert
    to that.
    """

    # NOTE(mdegans) this could probably use some common setup and teardown

    def test_out_resolution(self):
        """
        Test the out resolution is read from the file properly
        """
        for fn in CONFIG_FILES:
            with self.subTest(fn):
                master_cfg = sd.core.ConfigEngine(fn)
                # assumption: the resolution in all config files is 640x480
                #  todo: make a simple function to generate a config
                config = sd.detectors.deepstream.GstConfig(master_cfg, [dict(),], [dict(),])
                self.assertEqual(
                    config.out_resolution,
                    (640, 480),
                )

    def test_rows_and_columns(self):
        # TODO(mdegans): write better test
        master_cfg = sd.core.ConfigEngine(CONFIG_FILES[0])
        infer_configs = [dict(),]
        with self.subTest('one source == one row and column'):
            sources = [dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.rows_and_columns,
                1
            )
        with self.subTest('three sources == two rows and columns'):
            sources = [dict(), dict(), dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.rows_and_columns,
                2,
            )
        with self.subTest('five sources == three rows and columns'):
            sources = [dict(), dict(), dict(), dict(), dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.rows_and_columns,
                3
            )

    def test_tile_resolution(self):
        # TODO(mdegans): write better test
        # this also tests out_resolution
        master_cfg = sd.core.ConfigEngine(CONFIG_FILES[0])
        infer_configs = [dict(),]
        with self.subTest('one source == 640x480'):
            sources = [dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.tile_resolution,
                (640, 480),
            )
        with self.subTest('three sources == 320x240'):
            sources = [dict(), dict(), dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.tile_resolution,
                (320, 240),
            )
        with self.subTest('five sources == 213x160'):
            sources = [dict(), dict(), dict(), dict(), dict(),]
            config = sd.detectors.deepstream.GstConfig(
                master_cfg, infer_configs, sources)
            self.assertEqual(
                config.tile_resolution,
                (213, 160),
            )

    def test_host(self):
        master_cfg = sd.core.ConfigEngine(CONFIG_FILES[0])
        infer_configs = [dict(),]
        sources = [dict(),]
        config = sd.detectors.deepstream.GstConfig(
            master_cfg, infer_configs, sources)
        self.assertEqual(
            config.host, '0.0.0.0')

    def test_port(self):
        master_cfg = sd.core.ConfigEngine(CONFIG_FILES[0])
        infer_configs = [dict(),]
        sources = [dict(),]
        config = sd.detectors.deepstream.GstConfig(
            master_cfg, infer_configs, sources)
        self.assertEqual(
            config.port, 8000)

if __name__ == "__main__":
    unittest.main(verbosity=3)
