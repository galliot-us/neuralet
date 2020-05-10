import _init_path
from utils import export_keras_model_to_pb
from absl import flags
import tensorflow as tf


flags.DEFINE_string("export_path", 'inference_model', 'Should be an empty directory in which .pb file will be exported')
flags.DEFINE_string("model_path", "saved_model/model.h5", "Save directory")

FLAGS = tf.flags.FLAGS


def main(_):
    # Export protobuf model form h5 file.
    # e.g.: python export_inference_graph.py --model_path "saved_model/model.h5" --export_path "inference_model"
    export_keras_model_to_pb(FLAGS.model_path, FLAGS.export_path)


if __name__ == '__main__':
    tf.app.run(main)
