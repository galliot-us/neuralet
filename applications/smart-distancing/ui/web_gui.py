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

        return app


    def _run(self):
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
