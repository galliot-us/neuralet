from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
import keras
import os
from absl import flags
import tensorflow as tf
import _init_path
from utils import data_reader, plot_confusion_matrix, plot_results, bar_progress
import logging
import numpy as np
import wget

flags.DEFINE_string(
    "train_dir",
    "/work/tensorflow/face_mask_classifier/tpu/dataset/train",
    "The directory of train dataset",
)

flags.DEFINE_string(
    "validation_dir",
    "/work/tensorflow/face_mask_classifier/tpu/dataset/validation",
    "The directory of validation dataset",
)

flags.DEFINE_integer("input_size", 224, "The size of input image")

flags.DEFINE_integer("no_channels", 3, "Number of channels")

flags.DEFINE_integer("no_classes", 2, "The number of classes")

flags.DEFINE_string("save_dir", "saved_model", "Save directory")

flags.DEFINE_integer("epoch", 25, "Number of epochs to train")

flags.DEFINE_string("model_file_name", "model.h5", "Model .h5 name")

flags.DEFINE_float("learning_rate", 0.005, "Learning rate")

flags.DEFINE_integer("batch_size", 64, "Batch size")

flags.DEFINE_string("result_dir", "results", "Exported images path")

flags.DEFINE_list("classes", ["face", "face-mask"], "List of classes names")

flags.DEFINE_bool("pretraining", True, "True: Download the fine-tune model and retrain it")

FLAGS = tf.flags.FLAGS

if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)
    logging.info('{} folder is created because not exists'.format(FLAGS.save_dir))

if FLAGS.pretraining:
    url = 'https://github.com/neuralet/neuralet-models/blob/master/amd64/face_mask_classifier/model.h5?raw=true'
    if not os.path.isfile(os.path.join(FLAGS.save_dir, FLAGS.model_file_name)):
        print('model does not exist under: {} ,downloading from {}'.format(FLAGS.save_dir, url))
        wget.download(url, FLAGS.save_dir, bar_progress)


def main(_):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(FLAGS.input_size, FLAGS.input_size, FLAGS.no_channels), strides=(2, 2),
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
    if os.path.isfile(os.path.join(FLAGS.save_dir, FLAGS.model_file_name)):
        model.load_weights(os.path.join(FLAGS.save_dir, FLAGS.model_file_name))  # Load the saved model

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
        save_weights_only=False,
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

    plot_results(model, val_data_gen)

    # Plot confusion matrix
    logging.info("Start ploting confusion matrix")
    y_true = []
    y_probas = []
    for i in range(val_data_gen.samples):
        print('Getting prediction results is in progress {:.02f}%'.format(i / val_data_gen.samples * 100.0))
        input_img, label = val_data_gen.next()
        output = model.predict(input_img)
        y_true.append(np.argmax((label[0])))
        y_probas.append(np.argmax((output[0])))
    plot_confusion_matrix(y_true, y_probas)


if __name__ == "__main__":
    tf.app.run(main)
