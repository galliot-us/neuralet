"""
Add plugin nodes for custom layers of ssd-mobilenet-v1/v2 to the graph
and replace some newer operations which existing previous ones.
    
Reference:
   https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py
"""

import tensorrt as trt
import graphsurgeon as gs
import numpy as np

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
    """
    define a constant input tensor and set that as the input for the GridAnchor node 
    as UFF file does not provide an input element for the GridAnchor node
    """

    data = np.array([1, 1], dtype=np.float32) 
    anchor_input = gs.create_node("AnchorInput", "Const", value=data)  
    graph.append(anchor_input)
    graph.find_nodes_by_op("GridAnchor_TRT")[0].input.insert(0, "AnchorInput")

    return graph



def add_plugin_and_preprocess(graph, model, config):
    """add_plugin

    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """
    num_classes = int(config['MODEL']['NumberOfClasses'])
    min_size = float(config['MODEL']['MinSize']) 
    max_size = float(config['MODEL']['MaxSize'])
    input_order = [int(n) for n in config['MODEL']['InputOrder'].split(',')]
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
        minSize=min_size,  # was 0.2
        maxSize=max_size,  # was 0.95
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
        numClasses=num_classes,  # was 91
        inputOrder=input_order,
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
        "ToFloat": Input,
        "image_tensor": Input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
        "Concatenate": concat_priorbox, 
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)

    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")
    
    if ( "image_tensor:0" in graph.find_nodes_by_name("Input")[0].input ):
    # for ssd_mobilenet_v1
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")

    graph = replace_addv2(graph)
    graph = replace_fusedbnv3(graph)
    graph = parse_gridAnchor(graph)
    

    return graph
