"""Contains a Nvidia TensorRT based Detector and utilities."""
import ctypes
import logging
import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Required for initializing CUDA driver

import smart_distancing as sd
from smart_distancing.utils.fps_calculator import convert_infr_time_to_fps

from typing import (
    List,
    Sequence,
)

__all__ = [
    'JetsonDetector',
    'MobilenetSsdDetector',
]

logger = logging.getLogger(__name__)


def preprocess_trt(img):
    """Preprocess an image before TRT SSD inferencing."""
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = (2.0 / 255.0) * img - 1.0
    return img


class LoggingModuleTrtLogger(trt.Logger):
    """
    A TrtLogger that uses the standard python logging module internally.

    Attributes:
        SEVERITY_MAP (:obj:`dict` of :obj:`trt.Logger.SEVERITY`, int):
            a mapping from trt to python logging levels.
    """

    SEVERITY_MAP = {
        trt.Logger.INFO: logging.INFO,
        trt.Logger.VERBOSE: logging.DEBUG,
        trt.Logger.WARNING: logging.WARNING,
        trt.Logger.ERROR: logging.ERROR,
        trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
    }

    def log(self, severity, msg):
        logger.log(level=self.SEVERITY_MAP[severity], msg=msg)


class JetsonDetector(sd.detectors.BaseDetector):
    # TODO(mdegans): add context manager support to superclasss.
    #  Many of TensorRT's Objects are designed to operate as context managers
    #  and work safest when ther __enter__ and __exit__ are called with `with`...
    #  Many Detectors have similar needs.
    """
    Jetson sublcass of BaseDetector

    Args:
        config (:obj:`sd.core.ConfigEngine`): config.
        output_layout(int): todo: document what this does

    Attributes:
        logger (:obj:`LoggingModuleTrtLogger`):
            TensorRT logger that uses the python logging module.
        engine (:obj:`trt.ICudaEngine`):
            TensorRT engine.
        stream (:obj:`cuda.Stream`):
            CUDA stream to do work in.
        context (:obj:`trt.IExecutionContext`):
            TensorRT execution context.
    """

    PLATFORM = 'jetson'
    # TODO(mdegans): secure hash verification of all models
    DEFAULT_MODEL_URL = 'https://github.com/Tony607/jetson_nano_trt_tf_ssd/raw/master/packages/jetpack4.3/'

    engine = None
    stream = None
    fps = None
    # TODO(mdegans): Documentation says this is a singleton, so this should be fine,
    #  but it's still a good idea to test this.
    logger = LoggingModuleTrtLogger(trt.Logger.INFO)
    libflattenconcat_so = "/opt/libflattenconcat.so"

    def __init__(self, config, output_layout=7):
        self.output_layout = output_layout

        # lists for storing CUDA resources
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        # call the superclass __init__ which performs
        # common setup and calls self.load_model()
        super().__init__(config)

    def _load_engine(self) -> trt.ICudaEngine:
        """":return: a TensorRT engine from self.model_file."""
        logger.debug(f"loading engine: {self.model_file}")
        with open(self.model_file, 'rb') as f, trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if not engine:
                logger.warning("could not create engine")
            return engine

    def _load_plugins(self):
        """Required as Flattenconcat is not natively supported in TensorRT."""
        logger.debug(f"loading {self.libflattenconcat_so}")
        ctypes.CDLL(self.libflattenconcat_so)
        trt.init_libnvinfer_plugins(self.logger, '')

    def _create_context(self):
        """
        Create some space to store intermediate activation values. 
        Since the engine holds the network definition and trained parameters, additional space is necessary.
        """
        logger.debug("_create_context")
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                    self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        ctx = self.engine.create_execution_context()
        if not ctx:
            logger.warning("failed to create execution context")
        return ctx

    def load_model(self):
        """
        TensorRt Implemention of load_model()

        * load plugins - libflattenconcat.so
        * create self.engine - a new TensorRT engine
        * create self.stream - a new pycuda.device.Stream
        * create self.context - a new TensorRT execution context
        """
        self._load_plugins()
        self.engine = self._load_engine()
        self.stream = cuda.Stream()
        if not self.stream:
            logger.warning("could not create CUDA stream")
        self.context = self._create_context()

    def _postprocess_trt(self, img, output):
        """Postprocess TRT SSD output."""
        img_h, img_w, _ = img.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), self.output_layout):
            # index = int(output[prefix+0])
            conf = float(output[prefix + 2])
            if conf < float(self.score_threshold):
                continue
            x1 = (output[prefix + 3])  # * img_w)
            y1 = (output[prefix + 4])  # * img_h)
            x2 = (output[prefix + 5])  # * img_w)
            y2 = (output[prefix + 6])  # * img_h)
            cls = int(output[prefix + 1])
            boxes.append((y1, x1, y2, x2))
            confs.append(conf)
            clss.append(cls)
        return boxes, confs, clss

    def __del__(self):
        # NOTE(mdegans): because __del__ is not guaranteed in Python
        #  it might be a good idea to use a context manager instead like TensorRT itself.
        #  some libraries like aiohttp use __del__ a lot, but aiohttp is a buggy mess.
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def inference(self, img):
        """
        Detect objects in the input image. Calls on_frame() with the resulting detections.

        Args:
            img (:obj:`np.ndarray`): with shape (img_height, img_width, channels) (HWC)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score": s% }, {...}, {...}, ...]
        """
        img_resized = preprocess_trt(img)
        # transfer the data to the GPU, run inference and the copy the results back
        np.copyto(self.host_inputs[0], img_resized.ravel())

        # Start inference time
        t_begin = time.perf_counter()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        inference_time = time.perf_counter() - t_begin  # Second

        # Calculate Frames rate (fps)
        self._fps = convert_infr_time_to_fps(inference_time)
        output = self.host_outputs[0]
        boxes, scores, classes = self._postprocess_trt(img, output)
        result = []
        for i in range(len(boxes)):  # number of boxes
            if classes[i] == self.class_id + 1:
                result.append({"id": str(classes[i] - 1) + '-' + str(i), "bbox": boxes[i], "score": scores[i]})

        self.on_frame(result)

    @property
    def sources(self) -> List[str]:
        logger.warning(
            f"sources getter not implemented yet on {self.__class__.__name__}")
        return ['not implemented']

    @sources.setter
    def sources(self, sources: Sequence[str]):
        logger.warning(
            f"sources setter not implemented yet on {self.__class__.__name__}")

    @property
    def fps(self) -> int:
        return self._fps


class JetsonGstDetector(sd.detectors.BaseDetector):
    """Jetson GStreamer class (TODO(mdegans))"""


class MobilenetSsdDetector(JetsonDetector):
    """Mobilenet SSD implementation of JetsonDetector"""

    DEFAULT_MODEL_FILE = 'TRT_ssd_mobilenet_v2_coco.bin'
