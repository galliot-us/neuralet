import argparse
import os
import sys
from pathlib import Path

sys.path.append("../../")
from libs.config_engine import ConfigEngine

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def main(args):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.gfile.GFile(args.pipeline, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    config = ConfigEngine(args.config)
    pipeline_config.train_config.batch_size = int(config.get_section_dict('Student')['BatchSize'])
    data_dir = Path(config.get_section_dict('Teacher')['ImagePath']).parent
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(str(data_dir),
                                                                                            "train.record")]
    pipeline_config.eval_input_reader[:][0].tf_record_input_reader.input_path[:] = [os.path.join(str(data_dir),
                                                                                                 "val.record")]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.gfile.Open(args.pipeline, "wb") as f:
        f.write(config_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pipeline', required=True, type=str)
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    main(args)
