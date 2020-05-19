import os
import argparse
from pathlib import Path
import cv2 as cv


def extract_images_from_video(args):
    """
    This function extract each frame from the Oxford Town Center video file, resize it to a desired size and save them
    in the `images` directory in `jpg` format.
    User should specify the video path and the width and height of the desired output images.
    """
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
    while frame <= 4500:  # only first 4501 frames of Oxford Town Center Dataset has annotations.
        ret, img = video_cap.read()
        img = cv.resize(img, (img_width, img_height))
        image_path = os.path.join(images_dir, str(frame) + ".jpg")
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
