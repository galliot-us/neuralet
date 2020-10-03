"""build_engine.py

This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.

Input : spces of a ssd frozen inference graph in config.ini file
Output: TensorRT Engine file

Reference:
   https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py
"""


import os
import ctypes
import argparse
import configparser
import wget

import uff
import tensorrt as trt
import graphsurgeon as gs
import numpy as np
import add_plugin_and_preprocess_ssd_mobilenet as plugin

def main():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config.read(args.config)
    
    lib_flatten_concat_file = config['LIBFLATTENCONCAT']['Path']
    # initialize
    if trt.__version__[0] < '7':
        ctypes.CDLL(lib_flatten_concat_file)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    
    # compile the model into TensorRT engine
    model = config['MODEL']['Name'] 
    model_path = config['MODEL']['Input']
    url = config['MODEL']['DownloadPath']

    if not os.path.isfile(model_path):
        print('model does not exist under: ', model_path, 'downloading from ', url)
        wget.download(url, model_path)

    dynamic_graph = plugin.add_plugin_and_preprocess(
        gs.DynamicGraph(config['MODEL']['Input']),
        model,
        config)

    _ = uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        output_nodes=['NMS'],
        output_filename=config['MODEL']['TmpUff'],
        text=True,
        debug_mode=False)
    input_dims = tuple([int(x) for x in config['MODEL']['InputDims'].split(',')])
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', input_dims)
        parser.register_output('MarkOutput_0')
        parser.parse(config['MODEL']['TmpUff'], network)
        engine = builder.build_cuda_engine(network)
        
        buf = engine.serialize()
        with open(config['MODEL']['OutputBin'], 'wb') as f:
            f.write(buf)


if __name__ == '__main__':
    main()
