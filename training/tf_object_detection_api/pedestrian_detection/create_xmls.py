import os
import argparse
import pandas as pd
from pathlib import Path
from lxml import etree


def convert_to_xml(boxes, frame_number):
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "filename").text = str(frame_number) + ".jpg"
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = "1920"
    etree.SubElement(size, "height").text = "1080"
    etree.SubElement(size, "depth").text = "3"
    for bbox in boxes:
        obj = etree.SubElement(annotation, "object")
        etree.SubElement(obj, "name").text = "pedestrian"
        bounding_box = etree.SubElement(obj, "bndbox")
        etree.SubElement(bounding_box, "xmin").text = bbox[0]
        etree.SubElement(bounding_box, "ymin").text = bbox[1]
        etree.SubElement(bounding_box, "xmax").text = bbox[2]
        etree.SubElement(bounding_box, "ymax").text = bbox[3]
    xml = etree.ElementTree(annotation)
    return xml


def prepare_boxes(bbox):
    bbox[0] = str(int(max(0.0, bbox[0])))
    bbox[1] = str(int(max(0.0, bbox[1])))
    bbox[2] = str(int(min(1920, bbox[2])))
    bbox[3] = str(int(min(1080, bbox[3])))
    return bbox


def extract_labels(args):
    """
    This function will create xml annotation files from the Oxford Town Center csv annotation file
    your should specify the csv file path.
    The function will create a `xmls` directory and for each frame create a xml file in the directory
    """
    gt = pd.read_csv(args.annotation_path, header=None)
    annotation_path = Path(args.annotation_path)
    dataset_dir = annotation_path.parent
    labels_dir = os.path.join(str(dataset_dir), "xmls")
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    print("==================== Start Creating xml Files! ====================")
    for frame_number in range(4501):  # only first 4501 frames of Oxford Town Center Dataset has annotations.
        frame = gt[gt[1] == frame_number]
        x_min = list(frame[8])
        y_min = list(frame[9])
        x_max = list(frame[10])
        y_max = list(frame[11])
        bboxes = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in zip(x_min, y_min, x_max, y_max)]
        bboxes = list(map(prepare_boxes, bboxes))
        xml = convert_to_xml(bboxes, frame_number)

        xml_file_name = os.path.join(labels_dir, str(frame_number) + ".xml")
        xml.write(xml_file_name, pretty_print=True)
        print("annotation number {0} prepared".format(frame_number), end="\r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True, type=str)
    args = parser.parse_args()
    extract_labels(args)
