import os
import cv2
import numpy as np
import _init_paths


# Dataset from http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets
def video2im(src, test_path='../data/test_images', factor=2):
    """
    Extracts all test frames from a video and saves them as jpgs

    Args:
        src: The image source
        test_path: The path of images
        factor: Used for reducing the size of images by factor Ex: img_size = (width/factor, height/factor,3)

    Returns:

    """
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    if not os.path.isfile(src):
        print('The "%s" is not exist, run "download_sample_video.sh" script for downloading the dataset' % src)
        exit()
    frame = 0
    cap = cv2.VideoCapture(src)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total Frame Count:', length)

    while frame <= 4500:
        check, img = cap.read()
        if check:
            if frame >= 3700 and frame <= 4500:
                img = cv2.resize(img, (1920 // factor, 1080 // factor))
                cv2.imwrite(os.path.join(test_path, str(frame) + ".jpg"), img)
                print('Processed: ', frame, end='\r')
            frame += 1
        else:
            break
    print('test set is created successfully at "%s"' % test_path)
    cap.release()


if __name__ == '__main__':
    video2im('../data/TownCentreXVID.avi')
