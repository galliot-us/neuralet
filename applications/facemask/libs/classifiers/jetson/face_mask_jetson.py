import os
import numpy as np
import pathlib
import time
import wget
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from libs.utils.fps_calculator import convert_infr_time_to_fps


class Classifier:
    """
    Perform image classification with the given model. The model is a .trt file
    which if the classifier can not find it at the path it will download it
    from neuralet repository automatically.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.platform = self.config.DEVICE.split("-")[-1]
        # Frames Per Second
        self.fps = None
        if self.platform == "nano":
            print("<<<<<<<<<<<<<<<<<<<<<<<<<Warning>>>>>>>>>>>>>>>>>>>>>>>")
            print("The Face Detector of Jetson Nano is not implemented yet by dev team")
            self.model_name = "OFMClassifier_nano.trt"
        elif self.platform == "tx2":
            self.model_name = "OFMClassifier_tx2.trt"
        else:
            raise ValueError('Not supported device named: ', self.config.DEVICE)

        self.model_path = 'libs/classifiers/jetson/' + self.model_name
        if not os.path.isfile(self.model_path):
            url= "https://raw.githubusercontent.com/neuralet/neuralet-models/master/jetson-nano/OFMClassifier/" + self.model_name
            print("model does not exist under: ", self.model_path, "downloading from ", url)
            wget.download(url, self.model_path)
        

    def inference(self, resized_rgb_image) -> list:
        """
        Inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding class id output which is used for creating result
        Args:
            resized_rgb_image: Array of images with shape (no_images, img_height, img_width, channels)
        Returns:
            result: List of class id for each input image. ex: [0, 0, 1, 1, 0]
            scores: The classification confidence for each class. ex: [.99, .75, .80, 1.0]
        """
        self.INPUT_DATA_TYPE = np.float32
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(self.trt_logger)
        with open(self.model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.stream = cuda.Stream()
        
        self.host_in = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=self.INPUT_DATA_TYPE)
        self.host_out = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=self.INPUT_DATA_TYPE)
        self.devide_in = cuda.mem_alloc(self.host_in.nbytes)
        self.devide_out = cuda.mem_alloc(self.host_out.nbytes)


        if np.shape(resized_rgb_image)[0] == 0:
            return [], []
        result = []
        net_results = []
        for img in resized_rgb_image:
            img = np.expand_dims(img, axis=0)
            bindings = [int(self.devide_in), int(self.devide_out)]
            np.copyto(self.host_in, img.ravel())
            t_begin = time.perf_counter()
            cuda.memcpy_htod_async(self.devide_in, self.host_in, self.stream)
            self.context.execute_async(bindings=bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.host_out, self.devide_out, self.stream)
            self.stream.synchronize()
            inference_time = time.perf_counter() - t_begin  # Seconds
            self.fps = convert_infr_time_to_fps(inference_time)
            out = self.host_out
            pred = np.argmax(out)
            net_results.append(out)
            result.append(pred)

        # TODO: optimized without for
        scores = []
        for i, itm in enumerate(net_results):
            scores.append(itm[result[i]])

        return result, scores

