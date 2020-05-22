import numpy as np
import os


def export_results(nn_out, class_name, path, image_name):
    w = 300
    h = 300
    results = ''
    dt_fromat = "{0} {1} {2} {3} {4} {5} \n"
    for obj in nn_out:
        bbox = obj['bbox']
        bbox[0] = np.maximum(0, bbox[0])
        bbox[1] = np.maximum(0, bbox[1])
        bbox[2] = np.minimum(w, bbox[2])
        bbox[3] = np.minimum(h, bbox[3])
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        width = x1 - x0
        height = y1 - y0
        score = obj['score']
        results += dt_fromat.format(str(class_name), str(score), str(x0), str(y0), str(width),
                                    str(height))

        out_file = os.path.join(path, image_name + '.txt')
        with open(out_file, 'w') as file:
            file.write(results)
