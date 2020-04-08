"""ssd.py
This module implements the TrtSSD class.
"""
import ctypes
import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda

class Detector():
    
    def _load_plugins(self):
        ctypes.CDLL("/opt/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbinPath = 'libs/detectors/jetson/data/TRT_%s.bin' % self.model
        with open(TRTbinPath, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
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
        return self.engine.create_execution_context()

    def __init__(self, config, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.config = config
        self.model = self.config.get_section_dict('Detector')['Name']
        self.class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        self.conf_threshold = self.config.get_section_dict('Detector')['MinScore']
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def _preprocess_trt(self, img):
        """Preprocess an image before TRT SSD inferencing."""
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (2.0/255.0) * img - 1.0
        return img


    def _postprocess_trt(self, img, output):
        """Postprocess TRT SSD output."""
        img_h, img_w, _ = img.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), self.output_layout):
            #index = int(output[prefix+0])
            conf = float(output[prefix+2])
            if conf < float(self.conf_threshold):
                continue
            x1 = (output[prefix+3])# * img_w)
            y1 = (output[prefix+4])# * img_h)
            x2 = (output[prefix+5])# * img_w)
            y2 = (output[prefix+6])# * img_h)
            cls = int(output[prefix+1])
            boxes.append((y1, x1, y2, x2))
            confs.append(conf)
            clss.append(cls)
        return boxes, confs, clss


    def inference(self, img):
        """Detect objects in the input image."""
        img_resized = self._preprocess_trt(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())

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

        output = self.host_outputs[0]
        boxes, scores, classes = self._postprocess_trt(img, output)
        result = []
        for i in range(len(boxes)): #number of boxes
            if classes[i] == self.class_id+1:
                result.append({"id": str(classes[i]-1) + '-' + str(i), "bbox": boxes[i], "score": scores[i]})

        return result


