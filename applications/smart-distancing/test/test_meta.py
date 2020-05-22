import os
import sys
import unittest

# base paths, and import setup
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, PARENT_DIR)

import smart_distancing as sd


class TestMeta(unittest.TestCase):

    def test_meta_basic(self):
        b = sd.meta_pb2.BBox(
            left=123,
            top=555,
            height=99,
            width=999,
        )
        p = sd.meta_pb2.Person(
            uid=444,
            is_danger=True,
            danger_val=1.23,
            bbox=b,
        )
        f = sd.meta_pb2.Frame(
            frame_num=42,
            source_id=1234,
            people=[p,]
        )
        self.assertEqual(
            f.SerializeToString(),
            b'\x08*\x10\xd2\t\x1a\x16\x08\xbc\x03\x10\x01\x1d\xa4p\x9d?"\n\x08{\x10\xab\x04\x18c \xe7\x07',
        )

if __name__ == "__main__":
    unittest.main()
