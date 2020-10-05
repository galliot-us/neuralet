import threading
import cv2 as cv
import numpy as np
import time
from flask import Flask
from flask import render_template
from flask import Response
from libs.utils import visualization_utils as vis_util


class WebGUI:
    """
    The Webgui object implements a flask application and acts as an interface for users.
    Once it is created it will act as a central application for viewing outputs.

    :param config: Is a Config instance which provides necessary parameters.
    :param engine_instance:  A Config object which store all of the config parameters. Access to any parameter
        is possible by calling get_section_dict method.
    """

    def __init__(self, config, engine_instance):
        self.config = config
        self.__ENGINE_INSTANCE = engine_instance
        self._output_frame = None
        self._birds_view = None
        self._lock = threading.Lock()
        self._host = self.config.APP_HOST
        self._port = self.config.APP_PORT
        self.app = self.create_flask_app()
        self._displayed_items = {}  # all items here will be used at ui webpage

    def update(self, input_frame, nn_out):
        """
        Args:
            input_frame: uint8 numpy array with shape (img_height, img_width, 3)
            nn_out: List of dicionary contains normalized numbers of bounding boxes
            {'id' : '0-0', 'bbox' : [x0, y0, x1, y1], 'score' : 0.99(optional} of shape [N, 3] or [N, 2]

        Returns:
            draw the bounding boxes to an output frame
        """

        # Get a proper dictionary of bounding boxes and colors for visualizing_boxes_and_labels_on_image_array function
        output_dict = vis_util.visualization_preparation(nn_out)  # TODO
        #print(output_dict)
        category_index = {
            0: {
                "id": 0,
                "name": "Mask",
            },
            1: {
                "id": 1,
                "name": "Face",
            }}
        # Draw bounding boxes and other visualization factors on input_frame
        vis_util.visualize_boxes_and_labels_on_image_array(
            input_frame,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            output_dict["detection_colors"],
            category_index,
            instance_masks=output_dict.get("detection_masks"),
            use_normalized_coordinates=True,
            line_thickness=3,
        )
        # TODO: Implement perspective view for objects

        try:
            self._displayed_items['fps'] = self.__ENGINE_INSTANCE.detector.fps
        except:
            # fps is not implemented for the detector instance"
            self._displayed_items['fps'] = None

        # Put fps to the frame
        # region
        # -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_-
        txt_fps = 'Frames rate = ' + str(self._displayed_items['fps']) + '(fps)'  # Frames rate = 95 (fps)
        # (0, 0) is the top-left (x,y); normalized number between 0-1
        origin = (0.05, 0.93)
        vis_util.text_putter(input_frame, txt_fps, origin)
        # -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_-
        # endregion

        # Lock the main thread and copy input_frame to output_frame
        with self._lock:
            self._output_frame = input_frame.copy()

    def create_flask_app(self):
        # Create and return a flask instance named 'app'
        app = Flask(__name__)

        @app.route("/")
        def _index():
            # Render a html file located at templates as home page
            return render_template("index.html")

        @app.route("/video_feed")
        def video_feed():
            # Return the response generated along with the specific media
            # Type (mime type)
            return Response(
                self._generate(1), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        return app

    def _generate(self, out_frame: int):
        """
        Args:
            out_frame: The name of required frame. out_frame = 1 encoded camera/video frame otherwise
            encoded birds-eye window

        Returns:
            Yield and encode output_frame for flask the response object that is used by default in Flask
        """
        while True:
            with self._lock:
                # Check if the output frame is available, otherwise skip
                # The iteration of the loop
                if self._output_frame is None:
                    continue
                # Encode the frames in JPEG format
                (flag, encoded_input_img) = cv.imencode(".jpeg", self._output_frame)
                # Ensure the frame was successfully encoded
                if not flag:
                    continue

            # Yield the output frame in the byte format
            encoded_input_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_input_img) + b"\r\n")

            yield encoded_input_frame

    def _run(self):
        self.app.run(
            host=self._host, port=self._port, debug=True, threaded=True, use_reloader=False,
        )

    def start(self):
        """
        Start the thread's activity.
        It must be called at most once. It runes self._run method on a separate thread and starts
        process_video method at engine instance
        """
        threading.Thread(target=self._run).start()
        time.sleep(1)
        # Get video file path from the config
        video_path = self.config.APP_VIDEO_PATH
        self.__ENGINE_INSTANCE.process_video(video_path)
