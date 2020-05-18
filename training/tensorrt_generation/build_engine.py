"""build_engine.py

This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.

Input : ssd_mobilenet_v[1|2]_[coco|egohands].pb
Output: TRT_ssd_mobilenet_v[1|2]_[coco|egohands].bin

Reference:
   https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py
"""


import os
import ctypes
import argparse
import configparser

import uff
import tensorrt as trt
import graphsurgeon as gs
import numpy as np


def add_plugin(graph, model, config):
    """add_plugin

    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """
    numClasses = int(config['MODEL']['NumberOfClasses'])
    minSize = float(config['MODEL']['MinSize']) 
    maxSize = float(config['MODEL']['MaxSize'])
    inputOrder = [int(n) for n in config['MODEL']['InputOrder'].split(',')]# (config['MODEL']['InputOrder'])
    input_dims = tuple([int(x) for x in config['MODEL']['InputDims'].split(',')])

    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)
    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=(1,) + input_dims 
    )

    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=minSize,  # was 0.2
        maxSize=maxSize,  # was 0.95
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=numClasses,  # was 91
        inputOrder=inputOrder,
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )
    if trt.__version__[0] >= '6':
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT",
            axis=1,
            ignoreBatch=0
        )
    else:
        concat_box_loc = gs.create_plugin_node(
            "concat_box_loc",
            op="FlattenConcat_TRT"
        )
        concat_box_conf = gs.create_plugin_node(
            "concat_box_conf",
            op="FlattenConcat_TRT"
        )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "Cast": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
        "Concatenate": concat_priorbox,  # for other models
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    if model == 'ssd_mobilenet_v1_coco':
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

    return graph



def replace_addv2(graph):
    """Replace all 'AddV2' in the graph with 'Add'.
    NOTE: 'AddV2' is not supported by UFF parser.
    """
    for node in graph.find_nodes_by_op('AddV2'):
        gs.update_node(node, op='Add')
    return graph


def replace_fusedbnv3(graph):
    """Replace all 'FusedBatchNormV3' in the graph with 'FusedBatchNorm'.
    NOTE: 'FusedBatchNormV3' is not supported by UFF parser.
    https://devtalk.nvidia.com/default/topic/1066445/tensorrt/tensorrt-6-0-1-tensorflow-1-14-no-conversion-function-registered-for-layer-fusedbatchnormv3-yet/post/5403567/#5403567
    """
    for node in graph.find_nodes_by_op('FusedBatchNormV3'):
        gs.update_node(node, op='FusedBatchNorm')
    return graph

def parse_gridAnchor(graph):

    data = np.array([1, 1], dtype=np.float32) 
    anchor_input = gs.create_node("AnchorInput", "Const", value=data)  
    graph.append(anchor_input)
    graph.find_nodes_by_op("GridAnchor_TRT")[0].input.insert(0, "AnchorInput")

    return graph



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
    dynamic_graph = add_plugin(
        gs.DynamicGraph(config['MODEL']['Input']),
        model,
        config)
    dynamic_graph = replace_addv2(dynamic_graph)
    dynamic_graph = replace_fusedbnv3(dynamic_graph)
    dynamic_graph = parse_gridAnchor(dynamic_graph)
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
