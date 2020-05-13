import unittest
import os
import sys

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)


class TestPydsInstall(unittest.TestCase):

    PYDS_SO_ARM = '/opt/nvidia/deepstream/deepstream/sources/python/bindings/jetson/pyds.so'
    PYDS_SO_X86 = '/opt/nvidia/deepstream/deepstream/sources/python/bindings/x86-64/pyds.so'
    SO_PATHS = (PYDS_SO_ARM, PYDS_SO_X86)

    def test_pyds_import(self):
        if any(os.path.isfile(so) for so in self.SO_PATHS):
            from smart_distancing.detectors.deepstream._pyds import pyds
            self.assertTrue(hasattr(pyds, 'NVDS_USER_META'))
        else:
            self.skipTest(reason="pyds not installed")


if __name__ == "__main__":
    unittest.main()
