import threading
import time
import cv2 as cv
from flask import Flask
from flask import render_template
from flask import Response

from .utils import visualization_utils as vis_util

category_index ={0:{
    "id": 0,
    "name": "Pedestrian",
}}  # TODO: json file for detector config


class WebGUI:
    """
    The webgui object implements a flask application and acts as an interface for users.
    Once it is created it will act as a central application for the view outputs.

    :param config:
    :param engine_instance:
    """
    
    def __init__(self, config, engine_instance):
        self.config = config
        self.__ENGINE_INSTANCE = engine_instance
        self._output_frame = None
        self._lock = threading.Lock()
        self._host = self.config.get_section_dict("App")["Host"]
        self._port = int(self.config.get_section_dict("App")["Port"])
        self.app = self.create_flask_app()
        self._dist_threshold = float(self.config.get_section_dict("Detector")["DistThreshold"])
    
    def update(self, input_frame, nn_out, distances):
        """
        Args:
            input_frame: uint8 numpy array with shape (img_height, img_width, 3)
            nn_out: a list of dicionary contains normalized numbers of bounding boxes 
            {'id' : '0-0', 'bbox' : [x0, y0, x1, y1], 'score' : 0.99(optional} of shape [N, 3] or [N, 2]
            distances: a symmetric matrix of normalized distances

        Returns:
            draw the bounding boxes to an output frame
        """
        output_dict = vis_util.visualization_preparation(nn_out, distances, self._dist_threshold)
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
        with self._lock:
            self._output_frame = input_frame.copy()

    def create_flask_app(self):
        app = Flask(__name__)

        @app.route("/")
        def _index():
            return render_template("index.html")

        @app.route("/video_feed")
        def video_feed():
            # return the response generated along with the specific media
            # type (mime type)
            return Response(
                self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        return app

    def _generate(self):
        while True:
            with self._lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self._output_frame is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv.imencode(".jpeg", self._output_frame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )

    def _run(self):
        self.app.run(
                host=self._host, port=self._port, debug=True, threaded=True, use_reloader=False,
        )

    def start(self):
        threading.Thread(target=self._run).start()
        time.sleep(1)
        video_path = self.config.get_section_dict("App")["VideoPath"]
        self.__ENGINE_INSTANCE.process_video(video_path)
