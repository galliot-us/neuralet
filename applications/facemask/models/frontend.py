from models.backend import OFMClassifier
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import tensorflow.compat.v1 as tf


class FacemaskClassifierModel:
    def __init__(self, cfg):
        self._cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.channels = cfg.NO_CHANNEL
        input_image = Input(shape=(self.input_size, self.input_size, self.channels))

        nn = OFMClassifier(cfg)
        features = nn.model(input_image)
        out = Flatten()(features)
        out = Dense(10, activation='relu', name='fc_1')(out)
        out = Dense(cfg.NO_CLASSES, activation="softmax", name="fc_out")(out)

        self.model = Model(input_image, out)

        regularizer = regularizers.l2(1e-3)

        for layer in self.model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
