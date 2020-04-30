import xml.etree.ElementTree as ET
import os
import argparse
import numpy as np


def read_content(xml_file: str):
    """
    This function get the directory of .xml files and will extract annotations from them
    Args:
        xml_file: The directory of .xml files. Each xml file represents an image and its annotations

    Returns:
        filename: The name of image. e.g. 4500.jpg
        list_with_all_boxes: List of bounding boxes
        list_with_classes_name: List of objects class name
        width: The width that the image has
        height: The height that the image has

    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_classes_name = []
    # Get the image file name e.g. 4500.jpg
    filename = root.find("filename").text
    # The width and height of the image at xml file
    width = None
    height = None
    # Extracts the size of image
    for sz in root.iter("size"):
        width = sz.find("width").text
        height = sz.find("height").text

    # Extracts the class name and bounding boxes. An image may has more than one object\
    # so we should check all boxes and extract them
    for boxes in root.iter("object"):
        class_name = boxes.find("name").text
        ymin, xmin, ymax, xmax = None, None, None, None

        #  (x-top left, y-top left,x-bottom right, y-bottom right)
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_classes_name.append(class_name)

    return (
        filename,
        list_with_all_boxes,
        list_with_classes_name,
        int(width),
        int(height),
    )


def create_annot_custom_format(args):
    """
    This function gets the xml directory and extracts the ground truth from xml files and saves the output files
    Args:
        args: Contains two argument xml_dir: The directory of xml files and output_dir the directory for saving results

    Returns:

    """
    # Get xml_dir and output_dir from args
    path = args.xml_dir
    output = args.output_dir
    print("Start to extract ground truth...")
    # Lists the files at xml_directory (path)
    for filename in os.listdir(path):
        # Just consider xml files
        if not filename.endswith(".xml"):
            continue
        # Create full path of the xml file
        fullname = os.path.join(path, filename)
        # Extract filename, bounding boxes, classes_name, and image size information at the xml file
        name, boxes, classes, img_width, img_height = read_content(fullname)
        # Get the file name without its postfix. E.g. 4500.jpg -> 4500
        image_name = name.split(".")[0]
        annot = ""
        gt_fromat = "{0} {1} {2} {3} {4} \n"  # Class_name x, y, w, h
        for i, box in enumerate(boxes):
            # Do modifications
            box[0] = np.maximum(0, box[0])
            box[1] = np.maximum(0, box[1])
            box[2] = np.minimum(img_width, box[2])
            box[3] = np.minimum(img_height, box[3])
            bx_width = box[2] - box[0]
            bx_height = box[3] - box[1]
            # Each line of the annot represents a ground truth bounding box
            # (bounding boxes that a detector should detect)
            annot += gt_fromat.format(
                classes[i], str(box[0]), str(box[1]), str(bx_width), str(bx_height)
            )

        # Create a separate file for each xml file
        out_file = os.path.join(output, image_name + ".txt")
        with open(out_file, "w") as file:
            file.write(annot)
        print('The ground truth is successfully exported at "%s"' % output)


if __name__ == "__main__":
    # python create_annot_xmls.py --xml_dir xmls/ --output_dir results
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Make sure that the output directory exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    create_annot_custom_format(args)
