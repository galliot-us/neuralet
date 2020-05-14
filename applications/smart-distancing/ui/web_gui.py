import threading
import time
import cv2 as cv
import numpy as np
from datetime import date
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse
import uvicorn

from .utils import visualization_utils as vis_util
from tools.objects_post_process import extract_violating_objects
from tools.environment_score import mx_environment_scoring_consider_crowd


class WebGUI:
    """
    The Webgui object implements a fastapi application and acts as an interface for users.
    Once it is created it will act as a central application for viewing outputs.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    :param engine_instance:  A ConfigEngine object which store all of the config parameters. Access to any parameter
        is possible by calling get_section_dict method.
    """

    def __init__(self, config, engine_instance):
        self.config = config
        self.__ENGINE_INSTANCE = engine_instance
        self._output_frame = None
        self._birds_view = None
        self._lock = threading.Lock()
        self._host = self.config.get_section_dict("App")["Host"]
        self._port = int(self.config.get_section_dict("App")["Port"])
        self.app = self.create_fastapi_app()
        self._dist_threshold = float(self.config.get_section_dict("PostProcessor")["DistThreshold"])
        self._displayed_items = {}  # all items here will be used at ui webpage

        # TODO: read from config file
        file_name = str(date.today()) + '.csv'
        self.objects_log = './static/data/objects_log/' + file_name

    def update(self, input_frame, nn_out, distances):
        """
        Args:
            input_frame: uint8 numpy array with shape (img_height, img_width, 3)
            nn_out: List of dicionary contains normalized numbers of bounding boxes
            {'id' : '0-0', 'bbox' : [x0, y0, x1, y1], 'score' : 0.99(optional} of shape [N, 3] or [N, 2]
            distances: a symmetric matrix of normalized distances

        Returns:
            draw the bounding boxes to an output frame
        """
        # Create a black window for birds' eye view the size of window is constant (300, 200, 3)
        birds_eye_window = np.zeros((300, 200, 3), dtype="uint8")
        # Get a proper dictionary of bounding boxes and colors for visualizing_boxes_and_labels_on_image_array function
        output_dict = vis_util.visualization_preparation(nn_out, distances, self._dist_threshold)

        class_id = int(self.config.get_section_dict('Detector')['ClassID'])

        category_index = {class_id: {
            "id": class_id,
            "name": "Pedestrian",
        }}  # TODO: json file for detector config
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
        birds_eye_window = vis_util.birds_eye_view(birds_eye_window, output_dict["detection_boxes"],
                                                   output_dict["violating_objects"])
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

        # Put environment score to the frame
        # region
        # -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_-
        violating_objects = extract_violating_objects(distances, self._dist_threshold)
        env_score = mx_environment_scoring_consider_crowd(len(nn_out), len(violating_objects))
        txt_env_score = 'Env Score = ' + str(env_score)  # Env Score = 0.7
        origin = (0.05, 0.98)
        vis_util.text_putter(input_frame, txt_env_score, origin)
        # -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_- -_-
        # endregion

        # Lock the main thread and copy input_frame to output_frame
        with self._lock:
            self._output_frame = input_frame.copy()
            self._birds_view = birds_eye_window.copy()

    def create_fastapi_app(self):
        # Create and return a fastapi instance
        app = FastAPI()

        app.mount("/panel/static", StaticFiles(directory="/srv/frontend/static"), name="panel")
        app.mount("/static", StaticFiles(directory="ui/static"), name="static")

        @app.get("/panel/")
        async def panel():
            return FileResponse("/srv/frontend/index.html")

        @app.get("/")
        async def index():
            return RedirectResponse("/panel/")

        @app.get("/video_feed")
        def video_feed():
            # Return the response generated along with the specific media
            # Type (mime type)
            return StreamingResponse(
                self._generate(1), media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @app.get("/birds_view_feed")
        def birds_view_feed():
            # Return the response generated along with the specific media
            # Type (mime type)
            return StreamingResponse(
                self._generate(2), media_type="multipart/x-mixed-replace; boundary=frame"
            )

        return app

    def _generate(self, out_frame: int):
        """
        Args:
            out_frame: The name of required frame. out_frame = 1 encoded camera/video frame otherwise
            encoded birds-eye window

        Returns:
            Yield and encode output_frame the response object that is used by default in fastapi
        """
        while True:
            with self._lock:
                # Check if the output frame is available, otherwise skip
                # The iteration of the loop
                if self._output_frame is None:
                    continue
                # Encode the frames in JPEG format
                (flag, encoded_birds_eye_img) = cv.imencode(".jpeg", self._birds_view)
                (flag, encoded_input_img) = cv.imencode(".jpeg", self._output_frame)
                # Ensure the frame was successfully encoded
                if not flag:
                    continue

            # Yield the output frame in the byte format
            encoded_input_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_input_img) + b"\r\n")

            encoded_birds_eye_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_birds_eye_img) + b"\r\n")

            yield encoded_input_frame if out_frame == 1 else encoded_birds_eye_frame

    def _run(self):
        time.sleep(1)
        # Get video file path from the config
        video_path = self.config.get_section_dict("App")["VideoPath"]
        self.__ENGINE_INSTANCE.process_video(video_path)

    def start(self):
        """
        Start the thread's activity.
        It must be called at most once. It runes self._run method on a separate thread and starts
        process_video method at engine instance
        """
        process_thread = threading.Thread(target=self._run)
        process_thread.start()
        uvicorn.run(self.app, host=self._host, port=self._port, log_level='info')
        self.__ENGINE_INSTANCE.running_video = False
