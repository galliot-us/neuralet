from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras
import os
import matplotlib.pyplot as plt
from absl import flags
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
import logging
import numpy as np

flags.DEFINE_string(
    "train_dir",
    "/media/mehran/My files/My-Files/Company/Neuralet/object-detection/classifier_dataset_creators/classifier_data/dataset/train",
    "The directory of train dataset",
)

flags.DEFINE_string(
    "validation_dir",
    "/media/mehran/My files/My-Files/Company/Neuralet/object-detection/classifier_dataset_creators/classifier_data/dataset/validation",
    "The directory of validation dataset",
)

flags.DEFINE_integer("input_size", 224, "The size of input image")

flags.DEFINE_integer("no_channels", 3, "Number of channels")

flags.DEFINE_integer("no_classes", 2, "The number of classes")

flags.DEFINE_string("save_dir", "saved_model", "Save directory")

flags.DEFINE_string("export_dir", 'inference_model', 'Should be an empty directory in which .pb file will be exported')

flags.DEFINE_integer("epoch", 1, "Number of epochs to train")

flags.DEFINE_string("model_file_name", "model.h5", "Model .h5 name")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.DEFINE_integer("batch_size", 64, "Batch size")

flags.DEFINE_string("result_dir", "results", "Exported images path")

flags.DEFINE_list("classes", ["face", "mask-face"], "List of classes names")

FLAGS = tf.flags.FLAGS

if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)
    logging.info('{} folder is created because not exists'.format(FLAGS.save_dir))


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


def export_h5_to_pb(keras_model, export_path):
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


def plot_output(model, data_gen):
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



def main(_):
    model = Sequential()
    model.add(
        Conv2D(16, (3, 3), input_shape=(FLAGS.input_size, FLAGS.input_size, FLAGS.no_channels), strides=(2, 2),
               padding="same")
    )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(FLAGS.no_classes))
    model.add(Activation("sigmoid"))

    model.summary()

    # Create optimizer
    adam = optimizers.Adam(
        learning_rate=FLAGS.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False,
        epsilon=1e-08,
        decay=0.0,
    )

    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    cp_callback = keras.callbacks.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.save_dir, FLAGS.model_file_name),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        period=1,
    )

    train_data_gen = data_reader(
        path=FLAGS.train_dir, batch_size=FLAGS.batch_size, input_size=(FLAGS.input_size, FLAGS.input_size)
    )
    val_data_gen = data_reader(
        path=FLAGS.validation_dir, batch_size=1, input_size=(FLAGS.input_size, FLAGS.input_size)
    )
    # Start training
    logging.info("Start training ...")
    model.fit_generator(
        train_data_gen,
        steps_per_epoch=train_data_gen.samples // FLAGS.batch_size,
        epochs=FLAGS.epoch,
        validation_data=val_data_gen,
        validation_steps=val_data_gen.samples // 1,
        verbose=1,
        callbacks=[cp_callback],
    )

    plot_output(model, val_data_gen)

    export_h5_to_pb(keras_model=model, export_path=FLAGS.export_dir)
    logging.info("The inference model is exported at {}".format(FLAGS.save_dir))


if __name__ == "__main__":
    tf.app.run(main)
