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

class TestFindDeepstream(unittest.TestCase):

    def test_finds_correct_path(self):
        self.skipTest('because not implemented')
        # TODO(mdegans): write test
        # if jetpack 4.3, should return '4.0' and '.../deepstream-4.0'
        # if jetpack 4.4, should return '5.0' and '.../deepstream-5.0'
        # example:
        # >>> find_deepstream()
        # ('5.0', '/opt/nvidia/deepstream/deepstream-5.0')

if __name__ == "__main__":
    unittest.main(verbosity=3)
