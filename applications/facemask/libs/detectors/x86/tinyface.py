import numpy as np
import cv2
import time
import tensorflow as tf
from scipy.special import expit
import tiny_face_model
from libs.utils.fps_calculator import convert_infr_time_to_fps

weight_file_path = '../weights.pkl'
MAX_INPUT_DIM = 5000.0


class Detector:
    def __init__(self, config):
        self.config = config
        self.prob_thresh = 0.25
        self.fps = None
        self.score_final = None
        self.x = tf.placeholder(tf.float32, [1, None, None, 3])
        self.model = self.load_model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self._clusters = self.model.get_data_by_key("clusters")
        self._average_image = self.model.get_data_by_key("average_image")
    
    def load_model(self):
        model = tiny_face_model.Model(weight_file_path)
        self.score_final = model.tiny_face(self.x)
        return model

    def _calc_scales(self, raw_img):

        clusters_h = self._clusters[:, 3] - self._clusters[:, 1] + 1
        clusters_w = self._clusters[:, 2] - self._clusters[:, 0] + 1
        normal_idx = np.where(self._clusters[:, 4] == 1)
        raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
        min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                        np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
        scales_down = np.arange(min_scale, 0, 1.)
        scales_up = np.arange(0.5, max_scale, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)
        return scales

    def inference(self, resized_rgb_images):
        self.sess.run(tf.global_variables_initializer())

        scales = self._calc_scales(resized_rgb_images)
        bboxes = np.empty(shape=(0, 5))

        raw_img_f = resized_rgb_images.astype(np.float32)
        for s in scales:
            #print("Processing frame {} at scale {:.4f}".format(frame_count, s))
            img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            img = img - self._average_image
            img = img[np.newaxis, :]

            # we don't run every template on every scale ids of templates to ignore
            tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
            ignoredTids = list(set(range(0, self._clusters.shape[0])) - set(tids))

            # run through the net
            t_begin = time.perf_counter()
            score_final_tf = self.sess.run(self.score_final, feed_dict={self.x: img})
            inference_time = time.perf_counter() - t_begin  # Seconds
            self.fps = convert_infr_time_to_fps(inference_time)

            # collect scores
            score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
            prob_cls_tf = expit(score_cls_tf)
            prob_cls_tf[0, :, :, ignoredTids] = 0.0

            def _calc_bounding_boxes():
                # threshold for detection
                _, fy, fx, fc = np.where(prob_cls_tf > self.prob_thresh)

                # interpret heatmap into bounding boxes
                cy = fy * 8 - 1
                cx = fx * 8 - 1
                ch = self._clusters[fc, 3] - self._clusters[fc, 1] + 1
                cw = self._clusters[fc, 2] - self._clusters[fc, 0] + 1

                # extract bounding box refinement
                Nt = self._clusters.shape[0]
                tx = score_reg_tf[0, :, :, 0:Nt]
                ty = score_reg_tf[0, :, :, Nt:2 * Nt]
                tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
                th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

                # refine bounding boxes
                dcx = cw * tx[fy, fx, fc]
                dcy = ch * ty[fy, fx, fc]
                rcx = cx + dcx
                rcy = cy + dcy
                rcw = cw * np.exp(tw[fy, fx, fc])
                rch = ch * np.exp(th[fy, fx, fc])
                scores = prob_cls_tf[0, fy, fx, fc]
                tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                tmp_bboxes = tmp_bboxes.transpose()
                return tmp_bboxes

            tmp_bboxes = _calc_bounding_boxes()
            bboxes = np.vstack((bboxes, tmp_bboxes))  # <class 'tuple'>: (5265, 5)

        # non maximum suppression
        refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                  tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                  max_output_size=bboxes.shape[0], iou_threshold=0.1)
        refind_idx = self.sess.run(refind_idx)
        refined_bboxes = bboxes[refind_idx]
        nn_out = []
        boxes = []

        for r in refined_bboxes:
            _score = expit(r[4])
            _r = [int(x) for x in r[:4]]
            nn_out.append({"bbox": [np.abs(_r[0]), np.abs(_r[1]), np.abs(_r[2]), np.abs(_r[3])], "score": r[4]})
            # cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)
            boxes.append([_r[0], _r[1], _r[2], _r[3]])
        return nn_out
