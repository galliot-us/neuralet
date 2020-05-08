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

# where to find test data
TEST_DATA_DIR = os.path.join(THIS_DIR, 'data')
VIDEO_FILENAMES = (
    'TownCentreXVID.avi',
)


from smart_distancing.detectors.deepstream import _ds_engine
from smart_distancing.detectors.deepstream import _ds_config

class TestDsEngine(unittest.TestCase):

    def test_doctests(self):
        """test none of the doctests fail"""
        #FIXME(mdegans): this actually fails becuase the 'URI' is missing as
        # a source option. Docs need to be fixed. Also it should return an
        # error code in this case.
        self.assertEqual(
            doctest.testmod(_ds_engine, optionflags=doctest.ELLIPSIS)[0],
            0,
        )

    # config generator for uridecodebin
    FILE_SOURCE_CONFIGS = [
        {'uri': f'file://{fn}'} for fn in VIDEO_FILENAMES
    ]  # type: List[Dict[str, str]]
    # a list of paths to sample models
    SAMPLES_PATH = '/opt/nvidia/deepstream/deepstream/samples/models'
    MODEL_PATH =  os.path.join(SAMPLES_PATH, 'models')
    CONFIGS_PATH = os.path.join(SAMPLES_PATH, 'configs')
    DETECTOR_MODELS = (
        (
            'PrimaryDetector/resnet10.caffemodel',
            'PrimaryDetector/resnet10.prototxt',
            'PrimaryDetector/labels.txt',
            'PrimaryDetector/cal_trt.bin',  # int8 calibration data for supporting platforms
        ),
        (
            'PrimaryDetector_Nano/resnet10.caffemodel',
            'PrimaryDetector_Nano/resnet10.prototxt',
            'PrimaryDetector_Nano/labels.txt',
            None,  # int8 calibration data for supporting platforms
        ),
    )  # type: Tuple[str, str, str, Optional[str]]
    DETECTOR_CONFIGS = [
        {
            'model-file': m,
            'proto-file': p,
            'labelfile-path': l,
            'int8-calib-file': i
        } for m, p, l, i in DETECTOR_MODELS
    ]  # type: List[Dict[str, str]]
    CLASSIFIER_MODEL_DIRS = (
        'Secondary_CarColor',
        'Secondary_CarMake',
        'Secondary_VehicleTypes',
    )
    CLASSIFIER_MODELS = [
        [
            os.path.join(d, 'resnet18.caffemodel'),
            os.path.join(d, 'resnet18.prototxt'),
            os.path.join(d, 'labels.txt'),
            os.path.join(d, 'cal_trt.bin'),
            os.path.join(d, 'mean.ppm'),
        ] for d in CLASSIFIER_MODEL_DIRS
    ]
    # we test every primary model with every secondary model
    # with async mode both on and off
    CLASSIFIER_CONFIGS = []
    for detector_config in DETECTOR_CONFIGS:
        for async_mode in range(1):
            for c, p, l, i, m in CLASSIFIER_MODELS:
                conf = [
                    detector_config,  # the primary config
                    {
                        # oh fuck me, this can't be set dynamically.
                        # why is Nvidia in love with config files with craptastic absolute paths.
                        'model-file': m,
                        'proto-file': p,
                        'lablefile-path': l,
                        'int8-calib-file': i,
                        'mean-file': m, # bad file. very unfriendly (groans)
                        'classifier-async-mode': async_mode,
                    },
                ]
                CLASSIFIER_CONFIGS.append(conf)

    #FIXME(mdegans): right now this fails because some properties can't be set on nvinfer
    # without using an .ini file, so modifying DsEngine will be necessary.
    def test_start_stop(self):
        """test pipeline construct/destruct"""
        # this is copypasta from the TestGstEngine doctest
        single_detector_configs = [self.DETECTOR_CONFIGS[0],]
        single_file_configs = [self.FILE_SOURCE_CONFIGS[0],]

        config = _ds_config.DsConfig(single_detector_configs, single_file_configs)
        engine = _ds_engine.DsEngine(config)
        engine.start()
        engine.stop()
        engine.join(10)
        self.assertEqual(0, engine.exitcode)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
