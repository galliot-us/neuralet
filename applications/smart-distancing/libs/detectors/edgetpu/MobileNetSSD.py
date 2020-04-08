import sys
import os
import time
from threading import Thread
from queue import Queue
import numpy as np
import wget 

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

class Detector():
    def __init__(self, config):
        self.config = config

        self.model_name = self.config.get_section_dict('Detector')['Name']
        
        self.model_file = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        self.model_path = 'libs/detectors/edgetpu/data/' + self.model_file
        
        user_model_path = self.config.get_section_dict('Detector')['ModelPath'] 
        if len(user_model_path) > 0 :
            print('using %s as model' % user_model_path)
            self.model_path = user_model_path
        else:
            base_url = 'https://raw.githubusercontent.com/neuralet/neuralet-models/master/edge-tpu/'
            url = base_url + self.model_name + '/' + self.model_file

            if not os.path.isfile(self.model_path):
                print('model does not exist under: ' , self.model_path, 'downloading from ', url)
                wget.download(url, self.model_path)
    
        self.interpreter = Interpreter(self.model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        self.score_threshold = float(self.config.get_section_dict('Detector')['MinScore'])

    def inference(self, resized_rgb_image):
        input_image = np.expand_dims(resized_rgb_image, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_image)
        t_begin = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - t_begin
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        labels = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        num = self.interpreter.get_tensor(self.output_details[3]['index'])
        
        result = []
        for i in range(boxes.shape[1]): #number of boxes
            if labels[0, i] == self.class_id and scores[0, i] > self.score_threshold: 
                result.append({"id": str(self.class_id) + '-' + str(i), "bbox": boxes[0, i, :], "score": scores[0, i]})

        return result

