import numpy as np
import os


def export_results(nn_out, class_name, path, image_name, cv_img):
    img_resolution = cv_img.shape
    img_h, img_w = img_resolution[0], img_resolution[1]
    results = ''
    dt_fromat = "{0} {1} {2} {3} {4} {5} \n"
    for obj in nn_out:
        bbox = obj['bbox']
        bbox[0] = np.maximum(0, bbox[0])
        bbox[1] = np.maximum(0, bbox[1])
        bbox[2] = np.minimum(1, bbox[2])
        bbox[3] = np.minimum(1, bbox[3])
        x0 = bbox[1] * img_w
        y0 = bbox[0] * img_h
        x1 = bbox[3] * img_w
        y1 = bbox[2] * img_h
        width = x1 - x0
        height = y1 - y0
        score = obj['score']
        cls = class_name[int(float(obj["id"].split('-')[0]))]
        results += dt_fromat.format(str(cls), str(score), str(x0), str(y0), str(width),
                                    str(height))

        out_file = os.path.join(path, image_name + '.txt')
        with open(out_file, 'w') as file:
            file.write(results)
