# This Docker container will quantize a frozen graph object detection model and convert it to a quantized and tflite with TOCO tool. you can easily use this container by following these steps:
# 1. docker build -f toco-Dockerfile -t "neuralet/toco" .
# 2. docker run -v [PATH_TO_FROZEN_GRAPH_DIRECTORY]:/model_dir neuralet/toco --graph_def_file=[frozen graph file] 
# by running these command a detect.tflite file is created in the FROZEN_GRAPH_DIRECTORY that is a quantized version of the object detection model.
# you can also override other parameters like input_shapes

FROM tensorflow/tensorflow:1.15.0

VOLUME /model_dir

WORKDIR /model_dir

ENTRYPOINT ["/usr/local/bin/toco" ,"--output_format","TFLITE", "--input_shapes", "1,300,300,3","--input_arrays", "normalized_input_image_tensor" ,"--output_arrays","TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" ,"--inference_type" ,"QUANTIZED_UINT8" ,"--mean_values", "128" ,"--std_dev_values" ,"128","--change_concat_input_ranges", "false", "--allow_custom_op","--output_file","detect.tflite"]
