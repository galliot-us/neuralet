import xml.etree.ElementTree as ET
import os
import argparse


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_classes_name = []
    filename = root.find("filename").text
    width = None
    height = None
    for sz in root.iter("size"):
        width = sz.find("width").text
        height = sz.find("height").text

    for boxes in root.iter("object"):
        class_name = boxes.find("name").text
        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_classes_name.append(class_name)

    return filename, list_with_all_boxes, list_with_classes_name, int(width), int(height)


def create_annot_custom_format(args):
    path = args.xml_dir
    output = args.output_dir

    for filename in os.listdir(path):
        if not filename.endswith(".xml"):
            continue
        fullname = os.path.join(path, filename)
        name, boxes, classes, img_width, img_height = read_content(fullname)
        image_name = name.split(".")[0]
        annot = ''
        for i, box in enumerate(boxes):
            # Do modifications
            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[2] > img_width: box[2] = img_width
            if box[3] > img_height: box[3] = img_height

            bx_width = box[2] - box[0]
            bx_height = box[3] - box[1]
            annot += classes[i] + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(bx_width) + ' ' + str(
                bx_height) + '\n'  # x, y, w, h

        out_file = os.path.join(output, image_name + '.txt')
        with open(out_file, "w") as file:
            file.write(annot)


if __name__ == "__main__":
    # python create_annot_xmls.py --xml_dir xmls/ --output_dir results.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    create_annot_custom_format(args)
