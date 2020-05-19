import os
import argparse
from pathlib import Path
import cv2 as cv


def extract_images_from_video(args):
    video_path = Path(args.video_path)
    img_width = args.image_width
    img_height = args.image_height
    dataset_dir = video_path.parent
    images_dir = os.path.join(str(dataset_dir), "images")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    frame = 0
    video_cap = cv.VideoCapture(str(video_path))
    print("==================== Start Reading Video ! ====================")
    while frame <= 4500:
        ret, img = video_cap.read()
        img = cv.resize(img, (img_width, img_height))
        image_path = os.path.join(images_dir, str(frame)+".jpg")
        cv.imwrite(image_path, img)
        frame += 1
        print("frame number {0} extracted".format(frame), end="\r")
    video_cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--image_width", required=True, type=int)
    parser.add_argument("--image_height", required=True, type=int)
    args = parser.parse_args()
    extract_images_from_video(args)
