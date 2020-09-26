# This script is a modification of TensorFlow sample file. you can reach the file here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py
# license of original implementation is Apache V2
# ==============================================================================

r"""Convert the teacher generated dataset to TFRecord for student's training process and delete corresponding images and
annotations.
See: https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html
Example usage:
    python create_tfrecord.py \
        --data_dir $DATASET_DIR \
        --output_dir $DATASET_DIR \
        --label_map_path ./label_map.pbtxt \
        --validation_split 0.25
        --num_of_images_per_round 2500
"""

import glob
import hashlib
import io
import logging
import os
import random
import time

import PIL.Image
import tensorflow as tf
from lxml import etree
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_float('validation_split', 0.25, 'Percentage of Validation Set')
flags.DEFINE_integer('num_of_images_per_round', 100, 'number of frames for training in each round')

FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.
    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult_obj.append(int(0))
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            class_name = obj["name"]
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])
            truncated.append(int(0))
            poses.append('Unspecified'.encode('utf8'))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
    """Creates a TFRecord file from examples.
    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        xml_path = os.path.join(annotations_dir, str(example) + '.xml')

        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        try:
            tf_example = dict_to_tf_example(
                data,
                label_map_dict,
                image_dir)
            if tf_example:
                writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)
    writer.close()


def select_items(directory, num_of_images, max_retry=100):
    """
    select appropriate number of images from teacher's data directory and return list of selected images.
    Args:
        directory: directory to search
        num_of_images: how many images to select
        max_retry: how many times tries if the number of existing images on the directory are less than num_of_images

    Returns:
        a list that contains selected images name.
    """
    tries = 0
    while True:

        images_list = sorted(
            list(map(lambda x: int((x.split(".")[0]).split("/")[-1]), glob.glob(directory + "/*.jpg"))))
        print(directory)
        if len(images_list) >= num_of_images:
            # Wait a while for the data to be stored properly
            time.sleep(5)
            logging.info("{} images picked for training".format(num_of_images))
            return images_list[:num_of_images]
        else:
            if tries >= max_retry:
                raise StopIteration("The maximum retry has been reached")
            logging.info(
                "There is not {} images on the {} directory. try number {} from {}".format(num_of_images, directory,
                                                                                           tries + 1, max_retry))
            tries += 1
            time.sleep(300)


def main(_):
    data_dir = FLAGS.data_dir
    validation_split = FLAGS.validation_split
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Teacher Generated Dataset.')
    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'xmls')
    examples_list = select_items(image_dir, FLAGS.num_of_images_per_round)

    random.seed(42)
    num_examples = len(examples_list)
    num_train = int((1 - validation_split) * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    random.shuffle(train_examples)
    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    create_tf_record(
        train_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        train_examples)
    create_tf_record(
        val_output_path,
        label_map_dict,
        annotations_dir,
        image_dir,
        val_examples)

    for file_number in examples_list:
        os.remove(os.path.join(image_dir, str(file_number) + ".jpg"))
        os.remove(os.path.join(annotations_dir, str(file_number) + ".xml"))


if __name__ == '__main__':
    tf.app.run()
