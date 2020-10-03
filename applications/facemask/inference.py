from configs.config_handler import Config
import cv2 as cv
import PIL
import numpy as np


def main():
    config_path = 'configs/config.json'
    input_path = 'data/video/sample2.mov'
    output_path = ''

    cfg = Config(path=config_path)
    detector_input_size = (cfg.DETECTOR_INPUT_SIZE[0], cfg.DETECTOR_INPUT_SIZE[1], 3)
    classifier_img_size = (cfg.CLASSIFIER_INPUT_SIZE, cfg.CLASSIFIER_INPUT_SIZE, 3)

    device = cfg.DEVICE
    detector = None
    classifier = None
    output_vidwriter = None
    output_path = 'm.mp4'


    if device == "x86":
        from libs.detectors.x86.detector import Detector
        from libs.classifiers.x86.classifier import Classifier

    elif device == "EdgeTPU":
        from libs.detectors.edgetpu.detector import Detector
        from libs.classifiers.edgetpu.classifier import Classifier
    else:
        raise ValueError('Not supported device named: ', device)

    detector = Detector(cfg)
    classifier_model = Classifier(cfg)
    input_cap = cv.VideoCapture(input_path)
    
    while (input_cap.isOpened()):
        ret, raw_img = input_cap.read()
        if output_vidwriter is None:
            output_vidwriter = cv.VideoWriter(output_path, cv.VideoWriter_fourcc('M','J','P','G'), 24, (raw_img.shape[1],raw_img.shape[0]))
            height, width = raw_img.shape[:2]
        
        if ret == False:
            break
        _, cv_image = input_cap.read()
        if np.shape(cv_image) != ():
            resized_image = cv.resize(cv_image, tuple(detector_input_size[:2]))
            rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
            objects_list = detector.inference(rgb_resized_image)
            faces = []
            cordinates = []
            for obj in objects_list:
                if 'bbox' in obj.keys():
                    face_bbox = obj['bbox']  # [ymin, xmin, ymax, xmax]
                    xmin, xmax = np.multiply([face_bbox[1], face_bbox[3]], width)
                    ymin, ymax = np.multiply([face_bbox[0], face_bbox[2]], height)
                    croped_face = cv_image[int(ymin):int(ymin) + (int(ymax) - int(ymin)),
                                  int(xmin):int(xmin) + (int(xmax) - int(xmin))]
                    # Resizing input image
                    croped_face = cv.resize(croped_face, tuple(classifier_img_size[:2]))
                    croped_face = cv.cvtColor(croped_face, cv.COLOR_BGR2RGB)
                    # Normalizing input image to [0.0-1.0]
                    croped_face = croped_face / 255.0
                    faces.append(croped_face)
                    cordinates.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            
            faces = np.array(faces)
            face_mask_results, scores = classifier_model.inference(faces)
            for i, cor in enumerate(cordinates):
                if face_mask_results[i] == 1:
                    color = (0, 0 , 255)
                elif face_mask_results[i] == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 0)

                cv.rectangle(raw_img, (cor[0], cor[1]), (cor[2], cor[3]), color, 2)
            output_vidwriter.write(raw_img)
        else:
            continue

    input_cap.release()
    output_vidwriter.release()




if __name__ == '__main__':
    main()
