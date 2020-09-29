import tensorflow as tf
import numpy as np
import pathlib
import time
from libs.utils.fps_calculator import convert_infr_time_to_fps


def load_model(model_dir):
    """
    Args:
        model_name: Download the model based on its name and load the model
    Returns:
    """
    base_url = 'Not Available'
    # model_file = model_name + '.tar.gz'
    # model_dir = tf.keras.utils.get_file(
    #     fname=model_name,
    #     origin=base_url + model_file,
    #     untar=True)

    # model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir), None)
    print('Classifier model signatures: ', list(model.signatures.keys()))
    model = model.signatures[
        'predict']  # 'predict' signature was defined during exporting pb model change it if your signiture is someting else

    return model


class Classifier:
    """
    Perform image classification with the given model. The model is a protobuf
    file which if the classifier can not find it at the path it will download it
    from neuralet repository automatically.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.model_name = self.config.CLASSIFIER_MODEL_DIR
        self.classifier_model = load_model(self.model_name)
        # Frames Per Second
        self.fps = None

    def inference(self, resized_rgb_image: list) -> list:
        """
        Inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding class id output which is used for creating result
        Args:
            resized_rgb_image: List of images with shape (no_images, img_height, img_width, channels)
        Returns:
            result: List of class id for each input image [0, 0, 1, 1, 0]
        """
        if resized_rgb_image == []:
            return resized_rgb_image
        # iinput_image = np.expand_dims(resized_rgb_image, axis=0)
        input_tensor = tf.convert_to_tensor(resized_rgb_image, dtype=tf.float32)
        t_begin = time.perf_counter()
        output_dict = self.classifier_model(input_tensor)
        inference_time = time.perf_counter() - t_begin  # Seconds
        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)

        result = list(np.argmax(output_dict['scores'].numpy(), axis=1))  # returns class id
        return result
