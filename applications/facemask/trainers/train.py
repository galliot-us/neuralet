import os
import tensorflow as tf
from trainers.tensorboard_custom_callback import CustomTensorBoardCallback


class Train:
    def __init__(
            self, model, data_generator, cfg
    ):
        self._cfg = cfg
        self._model = model
        self._data_generator = data_generator

    def train(self):

        # float(self._cfg['train']['learning_rate'])
        lr = self._cfg.LEARNING_RATE

        opt = tf.keras.optimizers.Adam(
            learning_rate=lr,
            # rmsporp only:
            # centered=True
            # beta_1=0.9,
            # beta_2=0.999,
            # amsgrad=False,
            # epsilon=1e-08,
            # decay= 0.0002,
        )

        self._model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        train_data_gen = self._data_generator["train"]
        val_data_gen = self._data_generator["valid"]
        tensorboard_train_data = next(self._data_generator["train"])
        tensorboard_valid_data = next(self._data_generator["valid"])
        # todo implement at tools folder
        saveddir = os.path.join(self._cfg.SAVED_FOLDER, self._cfg.MODEL_NAME)
        if not os.path.exists(saveddir):
            os.makedirs(saveddir)
        log_dir = os.path.join(saveddir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            saveddir + '/model.h5',
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1
        )
        tb_callback = CustomTensorBoardCallback(train_data=tensorboard_train_data, val_data=tensorboard_valid_data,
                                                log_dir=log_dir, histogram_freq=0, num_of_images=24)

        self._model.fit_generator(
            train_data_gen,
            steps_per_epoch=train_data_gen.samples // self._cfg.BATCH_SIZE,
            epochs=self._cfg.EPOCHS,
            validation_data=val_data_gen,
            validation_steps=val_data_gen.samples // self._cfg.BATCH_SIZE,
            verbose=1,
            callbacks=[cp_callback, tb_callback]
        )
