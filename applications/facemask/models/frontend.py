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

model_backend_dic = {
    "OFMClassifier": OFMClassifier

}

def backend_loader(cfg):
    '''
    Load backend network based on the backend name at config

    Args:
        cfg: Is a Config instance which provides necessary parameters.
    Returns:
        backend: The backend network which is implemented at backend.py
    '''

    model_name = cfg.MODEL_NAME.split("-")[0]
    if model_name in model_backend_dic:
        backend = model_backend_dic.get(model_name)
    else:
        raise ValueError(f"{cfg.MODEL_NAME} is not in defined models")
    return backend(cfg)


class FacemaskClassifierModel:
    """
    Perform image classification with the given model backend name.
    :param cfg: Is a Config instance which provides necessary parameters.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.channels = cfg.NO_CHANNEL
        input_image = Input(shape=(self.input_size, self.input_size, self.channels))

        # load backend network
        backend_network = backend_loader(self._cfg)
        nn = backend_loader(self._cfg)
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
