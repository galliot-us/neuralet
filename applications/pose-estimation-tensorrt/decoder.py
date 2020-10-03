import numpy as np

from openpifpaf.data import COCO_KEYPOINTS, COCO_PERSON_SKELETON
from openpifpaf.decoder.pifpaf import PifPaf
from openpifpaf.functional import scalar_nonzero_clipped
from openpifpaf.decoder.utils import scalar_square_add_single

class PifPafDecoder():
    def __init__(self):
        self.seed_threshold = 0.2
        self.keypoint_threshold = 0.0
        self.instance_threshold = 0.1
        self._decode = self._factory_decode()

    def _factory_decode(self, extra_coupling=0.0,
                       experimental=False,
                       multi_scale=False,
                       multi_scale_hflip=True):
        """Configure PifPaf and Instantiate a decoder."""

        head_names = ('pif', 'paf', 'paf25')
        experimental = False 
        stride = 8 # TODO: model.head_strides[-1]

        pif_index = 0
        paf_index = 1
        pif_min_scale = 0.0
        paf_min_distance = 0.0
        paf_max_distance = None
        if multi_scale and multi_scale_hflip:
            resolutions = [1, 1.5, 2, 3, 5] * 2
            stride = [model.head_strides[-1] * r for r in resolutions]
            if not experimental:
                pif_index = [v * 3 for v in range(10)]
                paf_index = [v * 3 + 1 for v in range(10)]
            else:
                pif_index = [v * 2 for v in range(10)]
                paf_index = [v * 2 + 1 for v in range(10)]
            pif_min_scale = [0.0, 12.0, 16.0, 24.0, 40.0] * 2
            paf_min_distance = [v * 3.0 for v in pif_min_scale]
            paf_max_distance = [160.0, 240.0, 320.0, 480.0, None] * 2
            # paf_max_distance = [128.0, 192.0, 256.0, 384.0, None] * 2
        elif multi_scale and not multi_scale_hflip:
            resolutions = [1, 1.5, 2, 3, 5]
            stride = [model.head_strides[-1] * r for r in resolutions]
            if not experimental:
                pif_index = [v * 3 for v in range(5)]
                paf_index = [v * 3 + 1 for v in range(5)]
            else:
                pif_index = [v * 2 for v in range(5)]
                paf_index = [v * 2 + 1 for v in range(5)]
            pif_min_scale = [0.0, 12.0, 16.0, 24.0, 40.0]
            paf_min_distance = [v * 3.0 for v in pif_min_scale]
            paf_max_distance = [160.0, 240.0, 320.0, 480.0, None]
            # paf_max_distance = [128.0, 192.0, 256.0, 384.0, None]

        return PifPaf(
            stride,
            pif_index=pif_index,
            paf_index=paf_index,
            pif_min_scale=pif_min_scale,
            paf_min_distance=paf_min_distance,
            paf_max_distance=paf_max_distance,
            keypoints=COCO_KEYPOINTS,
            skeleton=COCO_PERSON_SKELETON,
            seed_threshold=self.seed_threshold
        )

    def decode(self, fields):
        return [self._annotations(field) for field in fields]

    def _annotations(self, fields, *, initial_annotations=None, meta=None, debug_image=None):
        annotations = self._decode(fields, initial_annotations=initial_annotations)

        # nms
        annotations = self.soft_nms(annotations)
        # threshold
        for ann in annotations:
            if meta is not None:
                self.suppress_outside_valid(ann, meta['valid_area'])
            kps = ann.data
            kps[kps[:, 2] < self.keypoint_threshold] = 0.0
        annotations = [ann for ann in annotations
                       if ann.score() >= self.instance_threshold]
        annotations = sorted(annotations, key=lambda a: -a.score())

        return annotations

    def soft_nms(self, annotations):
        if not annotations:
            return annotations

        occupied = np.zeros((
            len(annotations[0].data),
            int(max(np.max(ann.data[:, 1]) for ann in annotations) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in annotations) + 1),
        ), dtype=np.uint8)

        annotations = sorted(annotations, key=lambda a: -a.score())
        for ann in annotations:
            joint_scales = (np.maximum(4.0, ann.joint_scales)
                            if ann.joint_scales is not None
                            else np.ones((ann.data.shape[0]),) * 4.0)

            assert len(occupied) == len(ann.data)
            for xyv, occ, joint_s in zip(ann.data, occupied, joint_scales):
                v = xyv[2]
                if v == 0.0:
                    continue

                if scalar_nonzero_clipped(occ, xyv[0], xyv[1]):
                    xyv[2] = 0.0
                else:
                    scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)

        #if self.debug_visualizer is not None:
        #    LOG.debug('Occupied fields after NMS')
        #    self.debug_visualizer.occupied(occupied[0])
        #    self.debug_visualizer.occupied(occupied[4])

        annotations = [ann for ann in annotations if np.any(ann.data[:, 2] > 0.0)]
        annotations = sorted(annotations, key=lambda a: -a.score())
        return annotations


    def bbox_from_keypoints(self, kps):
        m = kps[:, 2] > 0
        if not np.any(m):
            return [0, 0, 0, 0]

        x, y = np.min(kps[:, 0][m]), np.min(kps[:, 1][m])
        w, h = np.max(kps[:, 0][m]) - x, np.max(kps[:, 1][m]) - y
        return [x, y, w, h]
