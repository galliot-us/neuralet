import numpy as np   
from libs.detectors.jetson.pose import PoseEstimator 
from libs.detectors.jetson.decoder import PifPafDecoder
import time
from libs.utils.fps_calculator import convert_infr_time_to_fps
import wget
import os


class Detector:
    """
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        # Get the model name from the config
        self.model_name = self.config.DETECTOR_NAME
        # Frames Per Second
        self.fps = None 
        self.w, self.h = (self.config.DETECTOR_INPUT_SIZE[0], self.config.DETECTOR_INPUT_SIZE[1])
        self.model_name = "resnet50-193-257.trt"
        self.model_path = 'libs/detectors/jetson/' + self.model_name
        if not os.path.isfile(self.model_path):
            url= "https://media.githubusercontent.com/media/neuralet/neuralet-models/master/jetson-tx2/pose-estimation-openpifpaf/" + self.model_name
            print("model does not exist under: ", self.model_path, "downloading from ", url)
            wget.download(url, self.model_path)

#        self.pose_estimator = PoseEstimator(self.model_path, (self.w, self.h))
#        self.decoder = PifPafDecoder()

    
    def inference(self, resized_rgb_image):
        """
        This method will perform inference and return the detected bounding boxes
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        assert resized_rgb_image.shape == (193, 257, 3)
        self.pose_estimator = PoseEstimator(self.model_path, (self.w, self.h))
        self.decoder = PifPafDecoder()
        t_begin = time.perf_counter()
        heads = self.pose_estimator.inference(resized_rgb_image)
        fields = [[field.cpu().numpy() for field in head] for head in heads]
        fields = [
            [[field[i] for field in head] for head in fields]
            for i in range(1) # 1 is image_batch.shape[0] which is num images in batch
            ]
        annotations = self.decoder.decode(fields)
        inference_time = time.perf_counter() - t_begin
        self.fps = convert_infr_time_to_fps(inference_time)
        result = []
        for l in annotations:
            for annotation_object in l:
                pred = annotation_object.data
                bbox_dict = {}
		# extracting face bounding box
                if np.all(pred[[0, 1, 2, 5, 6], -1] > 0.15):
                    x_min_face = int(pred[6, 0]) / self.w
                    x_max_face = int(pred[5, 0]) / self.w
                    y_max_face = int((pred[5, 1] + pred[6, 1]) / 2) / self.h
                    y_eyes = int((pred[1, 1] + pred[2, 1]) / 2) / self.h
                    y_min_face = 2 * y_eyes - y_max_face
                    if (y_max_face - y_min_face > 0) and (x_max_face - x_min_face > 0):
                        h_crop = y_max_face - y_min_face
                        x_min_face = max(0, x_min_face - 0.2 * h_crop)
                        y_min_face = max(0, y_min_face - 0.1 * h_crop)
                        x_max_face = min(self.w, x_min_face + 1 * h_crop)
                        y_max_face = min(self.h, y_min_face + 1 * h_crop)
                        bbox_dict["bbox"] = [y_min_face, x_min_face, y_max_face, x_max_face]
                        
                result.append(bbox_dict)
        del self.pose_estimator, self.decoder
        return result

