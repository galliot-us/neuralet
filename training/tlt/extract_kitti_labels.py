import os
import argparse
import pandas as pd
from pathlib import Path


def extract_labels(args):
    gt = pd.read_csv(args.annotation_path, header=None)
    width_ratio = args.image_width / 1920
    height_ratio = args.image_height / 1080
    annotation_path = Path(args.annotation_path)
    dataset_dir = annotation_path.parent
    labels_dir = os.path.join(str(dataset_dir), "labels")
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    def prepare_boxes(bbox):
        bbox[0] = str(round(max(0.0, bbox[0] * width_ratio), 2))
        bbox[1] = str(round(max(0.0, bbox[1] * height_ratio), 2))
        bbox[2] = str(round(min(float(args.image_width), bbox[2] * width_ratio), 2))
        bbox[3] = str(round(min(float(args.image_height), bbox[3] * height_ratio), 2))
        return bbox

    print("==================== Start Preparing Annotations in Kitti Format ! ====================")
    for frame_number in range(4501):
        frame = gt[gt[1] == frame_number]
        x_min = list(frame[8])
        y_min = list(frame[9])
        x_max = list(frame[10])
        y_max = list(frame[11])
        bboxes = [[xmin, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in zip(x_min, y_min, x_max, y_max)]
        bboxes = list(map(prepare_boxes, bboxes))
        with open(os.path.join(labels_dir, str(frame_number) + ".txt"), "w") as file:
            for box in bboxes:
                obj_data = ["pedestrian", "0.00", "0", "0.00"]
                obj_data.extend(box)
                obj_data = obj_data + ["0.00"] * 7
                file.write(" ".join(obj_data) + "\n")
        print("annotation number {0} prepared".format(frame_number), end="\r")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True, type=str)
    parser.add_argument("--image_width", required=True, type=int)
    parser.add_argument("--image_height", required=True, type=int)
    args = parser.parse_args()
    extract_labels(args)
