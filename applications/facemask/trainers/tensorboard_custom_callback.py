import tensorflow as tf
import keras
import io
from keras import backend as K
import numpy as np
from PIL import Image


class CustomTensorBoardCallback(keras.callbacks.TensorBoard):

    def __init__(self, train_data, val_data, num_of_images=8, **kwards):
        super(CustomTensorBoardCallback, self).__init__(**kwards)
        self.train_data = train_data
        self.val_data = val_data
        self.num_of_images = num_of_images
        # self.class_map = {0: "mask", 1: "not mask", 2: "not sure"}
        self.class_map = {0: "mask", 1: "not mask"}

    def make_image(self, numpy_img, gt, pred):
        height, width, channel = numpy_img.shape
        image = Image.fromarray(numpy_img)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

    def add_summary(self, img, gt, subset, i):
        image = (img * 255).astype("uint8")
        pred = np.squeeze(K.get_session().run(self.model.output, feed_dict={self.model.input: img[None, ...]}), axis=0)
        pred = int(np.argmax(pred))
        gt = int(np.argmax(gt))
        return tf.Summary.Value(tag='{}/{}/{}/{}'.format(subset, i, self.class_map[gt], self.class_map[pred]),
                                image=self.make_image(image, gt, pred))

    def on_epoch_end(self, epoch, logs):
        summary_str = []
        for i in range(self.val_data[0].shape[0]):
            val_summary = self.add_summary(self.val_data[0][i], self.val_data[1][i], "val", i)
            summary_str.append(val_summary)
            train_summary = self.add_summary(self.train_data[0][i], self.train_data[1][i], "train", i)
            summary_str.append(val_summary)
            summary_str.append(train_summary)
        self.writer.add_summary(tf.Summary(value=summary_str), global_step=epoch)
        return super().on_epoch_end(epoch, logs)
