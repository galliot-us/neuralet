import unittest
import doctest
import os
import sys

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)


from smart_distancing.detectors.deepstream import _ds_engine

class TestGstEngine(unittest.TestCase):

    def test_doctests(self):
        """test none of the doctests fail"""
        self.assertEqual(
            doctest.testmod(_ds_engine)[0],
            0,
        )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
