import os
import numpy as np
import pathlib
import time
import wget
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter
from libs.utils.fps_calculator import convert_infr_time_to_fps


class Classifier:
    """
    Perform image classification with the given model. The model is a .h5 file
    which if the classifier can not find it at the path it will download it
    from neuralet repository automatically.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.model_name = "OFMClassifier_edgetpu.tflite"
        if os.path.isfile(config.CLASSIFIER_MODEL_PATH):
            self.model_path = config.CLASSIFIER_MODEL_PATH
        else:
            self.model_path = 'data/classifiers/edgetpu/'
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)
            self.model_path = self.model_path + self.model_name

        # Frames Per Second
        self.fps = None
        if not os.path.isfile(self.model_path):
            url = "https://raw.githubusercontent.com/neuralet/neuralet-models/master/edge-tpu/OFMClassifier/OFMClassifier_edgetpu.tflite"
            print("model does not exist under: ", self.model_path, "downloading from ", url)
            wget.download(url, self.model_path)

        # Load TFLite model and allocate tensors
        self.interpreter = Interpreter(self.model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.interpreter.allocate_tensors()
        # Get the model input and output tensor details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

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
        if np.shape(resized_rgb_image)[0] == 0:
            return [], []
        resized_rgb_image = (resized_rgb_image * 255).astype("uint8")
        result = []
        net_results = []
        for img in resized_rgb_image:
            img = np.expand_dims(img, axis=0)
            self.interpreter.set_tensor(self.input_details[0]["index"], img)
            t_begin = time.perf_counter()
            self.interpreter.invoke()
            inference_time = time.perf_counter() - t_begin  # Second
            self.fps = convert_infr_time_to_fps(inference_time)
            net_output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            net_results.append(net_output)
            result.append(np.argmax(net_output))  # returns class id

        # TODO: optimized without for
        scores = []
        for i, itm in enumerate(net_results):
            scores.append(1)

        return result, scores
