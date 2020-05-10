import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from absl import flags
import os

flags.DEFINE_string("model_dir", "inference_model", "Directory of .pb file")

flags.DEFINE_integer("input_size", 224, "The size of input image")

flags.DEFINE_string("imgs_dir", "images", "Directroy of images for inference")

flags.DEFINE_list("classes", ["face", "face-mask"], "List of classes names")

flags.DEFINE_string("result_dir", "results", "Exported images path")

FLAGS = tf.flags.FLAGS


def main(_):
    face_mask_graph = tf.Graph()
    with tf.Session(graph=face_mask_graph) as sess:
        # Load protobuf model
        saved_metagraphdef = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir
        )

        # Get input tensor name
        tensor_name_input = (
            saved_metagraphdef.signature_def["predict"].inputs["images"].name
        )  # key_prediction, input_placeholder_name

        # Get output tensor dictionary
        tensor_name_output = {
            k: v.name
            for k, v in (saved_metagraphdef.signature_def["predict"].outputs.items())
        }

        # Get input and output tensors from the graph
        image_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name_input)
        out_tensor = tf.get_default_graph().get_tensor_by_name(
            tensor_name_output["scores"]
        )

        fig = plt.figure(figsize=(10, 10))

        for i, file_name in enumerate(os.listdir(FLAGS.imgs_dir)):
            if i < 25:
                plt.subplot(5, 5, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
            if file_name.endswith("jpg") or file_name.endswith(
                    "jpeg"
            ):  # Consider 'jpg' and 'jpeg' images
                img_path = os.path.join(FLAGS.imgs_dir, file_name)
                image = np.array(Image.open(img_path))
                image = (
                        image.astype(np.float32) / 255.0
                )  # Normalize image between [0-1]
                image = cv2.resize(
                    image, (FLAGS.input_size, FLAGS.input_size)
                )  # Resize the image based on input_size
                img = image.reshape(1, FLAGS.input_size, FLAGS.input_size, 3)
                out = sess.run(out_tensor, feed_dict={image_tensor: img})[0]
                class_id = np.argmax(out)
                plt.imshow(image, cmap=plt.cm.binary)
                plt.xlabel(FLAGS.classes[class_id])
    plt.show()
    if not os.path.exists(FLAGS.result_dir):
        os.mkdir(FLAGS.result_dir)
    fig.savefig(os.path.join(FLAGS.result_dir, 'inference_result.png'))


if __name__ == "__main__":
    tf.app.run(main)
