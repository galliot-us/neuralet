import tensorflow as tf
import numpy as np
import pathlib
import time
from libs.utils.fps_calculator import convert_infr_time_to_fps



class Classifier:
    """
    Perform image classification with the given model. The model is a protobuf
    file which if the classifier can not find it at the path it will download it
    from neuralet repository automatically.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, model, config):
        self.config = config
        self.model_dir = self.config.CLASSIFIER_MODEL_DIR
        self.classifier_model = model
        self.classifier_model.load_weights(self.model_dir)
        # Frames Per Second
        self.fps = None

    def inference(self, resized_rgb_image) -> list:
        """
        Inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding class id output which is used for creating result
        Args:
            resized_rgb_image: Array of images with shape (no_images, img_height, img_width, channels)
        Returns:
            result: List of class id for each input image [0, 0, 1, 1, 0]
        """
        if np.shape(resized_rgb_image)[0] == 0:
            return resized_rgb_image
        #input_image = np.expand_dims(resized_rgb_image, axis=0)
        t_begin = time.perf_counter()
        output_dict = self.classifier_model.predict(resized_rgb_image)
        inference_time = time.perf_counter() - t_begin  # Seconds
        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)
        result = list(np.argmax(output_dict, axis=1))  # returns class id
        return result
