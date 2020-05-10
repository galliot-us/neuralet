# Set up paths for the Face Mask Recognition
import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


currentPath = os.path.dirname(os.path.realpath(__file__))
libPath = currentPath
add_path(libPath)