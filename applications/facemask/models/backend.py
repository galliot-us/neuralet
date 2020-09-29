from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, BatchNormalization, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


class OFMClassifier:
    def __init__(self, cfg):
        self._input_size = cfg.INPUT_SIZE
        self._channels = cfg.NO_CHANNEL
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=1, padding="same", activation='relu',
                         input_shape=(self._input_size, self._input_size, self._channels)))
        model.add(Conv2D(32, (3, 3), strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation='relu',
                         input_shape=(self._input_size, self._input_size, self._channels)))
        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation='relu',
                         input_shape=(self._input_size, self._input_size, self._channels)))
        model.add(Conv2D(128, (3, 3), strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.2))

        model.add(Dense(64, activation='relu'))

        regularizer = regularizers.l2(1e-3)
        for layer in model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        input_image = Input(shape=(self._input_size, self._input_size, self._channels))
        x = model(input_image)
        self.model = Model(input_image, x)
        model.summary()
