import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np
import cv2


class DataLoader:
    def __init__(self, cfg):
        self._cfg = cfg

        self.total_train = 0
        self.total_valid = 0

        self._validation_dir = os.path.join(self._cfg.PATH, self._cfg.VALID_FOLDER)
        self._train_dir = os.path.join(self._cfg.PATH, self._cfg.TRAIN_FOLDER)

    def _preproc_func(self, image):
        result = image
        # if_noise = random.randint(0,3)
        # if if_noise == 0:
        #    result = self.add_noise(image)

        result = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return result

    def _preproc_func_test(self, image):
        result = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return result

    def add_noise(self, img):
        '''Add random noise to an image'''
        VARIABILITY = random.randint(0, 30)
        deviation = VARIABILITY * random.random()
        noise = np.random.normal(0, deviation, img.shape)
        img += noise
        np.clip(img, 0., 255.)
        img = cv2.convertScaleAbs(img)
        return np.float64(img)
        # return img

    def _keras_based_data_reader(self, **kwargs):
        directory = kwargs["data_directory"]
        if not kwargs["train"]:
            data_generator = ImageDataGenerator(rescale=1.0 / 255,
                                                preprocessing_function=self._preproc_func_test
                                                # color_mode="grayscale"
                                                )
            train_data_gen = data_generator.flow_from_directory(
                batch_size=self._cfg.BATCH_SIZE,
                directory=directory,
                shuffle=False,
                target_size=(self._cfg.INPUT_SIZE, self._cfg.INPUT_SIZE),
                class_mode="categorical",
                color_mode="rgb"
            )
        else:
            data_generator = ImageDataGenerator(rescale=1.0 / 255,
                                                # width_shift_range=[-.12, .12],
                                                # height_shift_range=[-.12, .12],
                                                # horizontal_flip=True,
                                                # rotation_range=10,
                                                # brightness_range=[.9, 1.1],
                                                # zoom_range=[.9, 1.1],
                                                # channel_shift_range=50.,
                                                preprocessing_function=self._preproc_func
                                                )

            train_data_gen = data_generator.flow_from_directory(
                batch_size=self._cfg.BATCH_SIZE,
                directory=directory,
                shuffle=True,
                target_size=(self._cfg.INPUT_SIZE, self._cfg.INPUT_SIZE),
                class_mode="categorical",
                color_mode="rgb"
            )
        print('--------------- class IDs ----------------- : ', train_data_gen.class_indices)
        return train_data_gen

    def _get_params(self, item):
        if item == "train":
            directory = self._train_dir
            epoch = self._cfg.EPOCHS
            is_train = True
        elif item == "valid":
            directory = self._validation_dir
            epoch = 1
            is_train = False
        else:
            raise Exception(
                f"{item} argument is not defined. Item should be 'train' or 'valid'"
            )
        return directory, epoch, is_train

    def __getitem__(self, item):
        directory, epoch, is_train = self._get_params(item)

        train_data_gen = self._keras_based_data_reader(
            data_directory=directory, epoch=epoch, train=is_train
        )

        return train_data_gen
