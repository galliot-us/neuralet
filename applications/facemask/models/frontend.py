from models.backend import OFMClassifier


class FacemaskClassifierModel:
    def __init__(self, cfg):
        self._cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.channels = cfg.NO_CHANNEL
        input_image = Input(shape=(self.input_size, self.input_size, self.channels))

        nn = OFMClassifier(cfg)
        features = nn.model(input_image)

        out = Dense(64, activation='relu', name='fc_1')(features)
        out = Dense(10, activation='relu', name='fc_2')(out)
        out = Dense(cfg.NO_CLASSES, activation="softmax", name="fc_out")(out)

        self.model = Model(input_image, out)

        regularizer = regularizers.l2(1e-3)

        for layer in self.model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
