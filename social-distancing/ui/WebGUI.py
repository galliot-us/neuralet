import threading
import time
import cv2 as cv
from flask import Flask
from flask import render_template
from flask import Response

from .utils import visualization_utils as vis_util

class WebGUI:

    def __init__(self, config, engine_instance):
        self.config = config
        self.__ENGINE_INSTANCE = engine_instance
        self._output_frame = None
        self._lock = threading.Lock()
        self.app = self.create_flask_app()

    def update(self, input_frame, boxes, img_shape, distances):
        # TODO: docstring after completing
        output_dict = vis_util.visualization_preparation(boxes, img_shape)
        vis_util.visualize_boxes_and_labels_on_image_array(
            input_frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3)
        with self._lock:
            self._output_frame = input_frame.copy()
            self._output_frame = cv.resize(self._output_frame, (220, 160))


    def create_flask_app(self):
        app = Flask(__name__)
        #@staticmethod
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
        # TODO: docstring after completing
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
            host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False,
        )

    def start(self):
        threading.Thread(target = self._run).start()
        time.sleep(1)
        video_path = self.config.get_section_dict('App')['VideoPath']
        self.__ENGINE_INSTANCE.process_video(video_path)
