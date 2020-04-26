# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
Most functions do not return a value, instead they modify the image itself.

"""
import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2 as cv

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

STANDARD_COLORS = [
    "Green",
    "Blue"
]


def draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=(255, 0, 0),  # RGB
        thickness=4,
        display_str_list=(),
        use_normalized_coordinates=True,
):
    """Adds a bounding box to an image (numpy array).
  
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
  
    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_bounding_box_on_image(
        image_pil,
        ymin,
        xmin,
        ymax,
        xmax,
        color,
        thickness,
        display_str_list,
        use_normalized_coordinates,
    )
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=(255, 0, 0),  # RGB
        thickness=4,
        display_str_list=(),
        use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.
  
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
  
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.
  
    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_keypoints_on_image_array(
        image, keypoints, color="red", radius=2, use_normalized_coordinates=True
):
    """Draws keypoints on an image (numpy array).
  
    Args:
      image: a numpy array with shape [height, width, 3].
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    draw_keypoints_on_image(
        image_pil, keypoints, color, radius, use_normalized_coordinates
    )
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(
        image, keypoints, color="red", radius=2, use_normalized_coordinates=True
):
    """Draws keypoints on an image.
  
    Args:
      image: a PIL.Image object.
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse(
            [
                (keypoint_x - radius, keypoint_y - radius),
                (keypoint_x + radius, keypoint_y + radius),
            ],
            outline=color,
            fill=color,
        )


def draw_mask_on_image_array(image, mask, color="red", alpha=0.4):
    """Draws mask on an image.
  
    Args:
      image: uint8 numpy array with shape (img_height, img_height, 3)
      mask: a uint8 numpy array of shape (img_height, img_height) with
        values between either 0 or 1.
      color: color to draw the keypoints with. Default is red.
      alpha: transparency value between 0 and 1. (default: 0.4)
  
    Raises:
      ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError("`image` not of type np.uint8")
    if mask.dtype != np.uint8:
        raise ValueError("`mask` not of type np.uint8")
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "The image has spatial dimensions %s but the mask has "
            "dimensions %s" % (image.shape[:2], mask.shape)
        )
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(
        list(rgb), [1, 1, 3]
    )
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert("RGB")))


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        colors,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=0.0,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color="black",
        skip_scores=False,
        skip_labels=False,
):
    """Overlay labeled boxes on an image with formatted scores and label names.
  
    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.
  
    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      colors: BGR fromat colors for drawing the boxes
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width] with
        values ranging between 0 and 1, can be None.
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
        with values ranging between 0 and 1, can be None.
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection
  
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ""
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]["name"]
                        else:
                            class_name = "N/A"
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = "{}%".format(int(100 * scores[i]))
                    else:
                        display_str = "{}: {}%".format(
                            display_str, int(100 * scores[i])
                        )
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = "DarkOrange"
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)
                        ]

    # Draw all boxes onto image.
    for box, color in zip(boxes, colors):
        xmin, ymin, xmax, ymax = box
        if instance_masks is not None:
            draw_mask_on_image_array(image, box_to_instance_masks_map[tuple(box)], color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[tuple(box)], color="red", alpha=1.0
            )
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[tuple(box)],
            use_normalized_coordinates=use_normalized_coordinates,
        )
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates,
            )

    return image


def visualization_preparation(nn_out, distances, dist_threshold):
    """
    prepare the objects boxes and id in order to visualize

    Args:
        nn_out: a list of dicionary contains normalized numbers of bonding boxes
        {'id' : '0-0', 'bbox' : [x0, y0, x1, y1], 'score' : 0.99(optional} of shape [N, 3] or [N, 2]
        distances: a symmetric matrix of normalized distances
        dist_threshold: the minimum distance for considering unsafe distance between objects
    Returns:
        an output dictionary contains object classes, boxes, scores
    """
    output_dict = {}
    detection_classes = []
    detection_scores = []
    detection_boxes = []
    is_violating = []
    colors = []
    
    distance = np.amin(distances + np.identity(len(distances)) * dist_threshold * 2, 0)
    for i, obj in enumerate(nn_out):
        # Colorizing bounding box based on the distances between them
        # R = 255 when dist=0 and R = 0 when dist > dist_threshold
        redness_factor = 1.5
        r_channel = np.maximum(255 * (dist_threshold - distance[i]) / dist_threshold, 0) * redness_factor
        g_channel = 255 - r_channel
        b_channel = 0
        # Create a tuple object of colors
        color = (int(b_channel), int(g_channel), int(r_channel))
        # Get the object id
        obj_id = obj["id"]
        # Split and get the first item of obj_id
        obj_id = obj_id.split("-")[0]
        box = obj["bbox"]
        if "score" in obj:
            score = obj["score"]
        else:
            score = 1.0
        # Append all processed items
        detection_classes.append(int(obj_id))
        detection_scores.append(score)
        detection_boxes.append(box)
        colors.append(color)
        is_violating.append(True) if distance[i] < dist_threshold else is_violating.append(False)
    output_dict["detection_boxes"] = np.array(detection_boxes)
    output_dict["detection_scores"] = detection_scores
    output_dict["detection_classes"] = detection_classes
    output_dict["violating_objects"] = is_violating
    output_dict["detection_colors"] = colors
    return output_dict


def birds_eye_view(input_frame, boxes, is_violating):
    """
    This function receives a black window and draw circles (based on boxes) at the frame.
    Args:
        input_frame: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: A numpy array of shape [N, 4]
        is_violating: List of boolean (True/False) which indicates the correspond object at boxes is
        a violating object or not

    Returns:
        input_frame: Frame with red and green circles

    """
    h, w = input_frame.shape[0:2]
    for i, box in enumerate(boxes):
        center_x = int((box[0] * w + box[2] * w) / 2)
        center_y = int((box[1] * h + box[3] * h) / 2)
        center_coordinates = (center_x, center_y)
        color = (0, 0, 255) if is_violating[i] else (0, 255, 0)
        input_frame = cv.circle(input_frame, center_coordinates, 2, color, 2)
    return input_frame


def text_putter(input_frame, txt, origin, fontscale=0.75, color=(255, 0, 20), thickness=2):
    """
    The function renders the specified text string in the image. This function does not return a
    value instead it modifies the input image.

    Args:
        input_frame: The source image, is an RGB image.
        txt: The specific text string for drawing.
        origin: Top-left corner of the text string in the image. The resolution should be normalized between 0-1
        fontscale: Font scale factor that is multiplied by the font-specific base size.
        color: Text Color. (BGR format)
        thickness: Thickness of the lines used to draw a text.
    """
    resolution = input_frame.shape
    origin = int(resolution[1] * origin[0]), int(resolution[0] * origin[1])
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(input_frame, txt, origin, font, fontscale,
               color, thickness, cv.LINE_AA)
