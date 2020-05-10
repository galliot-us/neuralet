from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS


def plot_confusion_matrix(cls_true, cls_pred):
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred,
                          )
    plt.ion()
    plt.matshow(cm, cmap=plt.cm.Reds)
    num_classes = 2
    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, FLAGS.classes)
    plt.yticks(tick_marks, FLAGS.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="black")
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()
    plt.savefig(os.path.join(FLAGS.result_dir, 'confusion_matrix.png'))


def data_reader(path, batch_size, input_size):
    """
    Using keras image generator to read images
    Args:
        path: Image path (each folder at this path will be considered as a class)
        batch_size: Batch size
        input_size: The size of the network input

    Returns:
        data: A generator of images

    """
    data_generator = ImageDataGenerator(rescale=1.0 / 255)
    data = data_generator.flow_from_directory(
        batch_size=batch_size,
        directory=path,
        shuffle=True,
        target_size=input_size,
        class_mode="categorical",
    )
    return data


def export_keras_model_to_pb(keras_model, export_path):
    """
    Function to export Keras model to Protocol Buffer format

    Args:
        keras_model: Keras Model instance
        export_path: Path to store Protocol Buffer model

    Returns:
    """
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)  # export_path must be an empty folder

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(inputs={"images": keras_model.input},
                                      outputs={"scores": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={"predict": signature})

    builder.save()


def plot_results(model, data_gen):
    """
    This function gets the output of the model for 25 random images from validation set and plots them
    Args:
        model: Keras model
        data_gen: Keras image generator

    Returns:

    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        x, y = data_gen.next()
        out = model.predict(x)

        x = np.reshape(x, newshape=(FLAGS.input_size, FLAGS.input_size, FLAGS.no_channels))
        x = x * 255
        x = x.astype('uint8')
        plt.imshow(x, cmap=plt.cm.binary)
        label_id = np.argmax(y)
        prd_id = np.argmax(out)
        plt.xlabel(FLAGS.classes[label_id] + ' / ' + FLAGS.classes[prd_id])
    plt.show()
    if not os.path.exists(FLAGS.result_dir):
        os.mkdir(FLAGS.result_dir)
    fig.savefig(os.path.join(FLAGS.result_dir, 'eval_results.png'))
