# (C) Michael de Gans, 2020
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import multiprocessing
import queue
import logging

# import gstreamer bidings
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import (
    Gst,
    GLib,
)
# import python deepstream
from smart_distancing.detectors.deepstream import pyds
# import config stuff
from smart_distancing.detectors.deepstream._ds_config import (
    GstConfig,
    DsConfig,
    ElemConfig,
)
# typing
from typing import (
    Any,
    Callable,
    Iterator,
    Iterable,
    Optional,
    List,
    Mapping,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from smart_distancing import Detections
else:
    Detections = None


__all__ = [
    'GstEngine',
    'PadProbeCallback',
    'link_many',
    'frame_meta_iterator',
    'obj_meta_iterator',
]

PadProbeCallback = Callable[
    [Gst.Pad, Gst.PadProbeInfo, Any],
    Gst.PadProbeReturn,
]
"""
Signature of Gsteamer Pad Probe Callback
"""


def link_many(elements: Iterable[Gst.Element]):
    """
    Link many Gst.Element.
    
    (linear, assumes Always Availability of src and sink pads).

    Returns:
        bool: False on failure, True on success.
    """
    elements = iter(elements)
    last = next(elements)
    for element in elements:
        if not last.link(element):
            return False
    return True


# these two functions below are used by DsEngine to parse pyds metadata

def frame_meta_iterator(frame_meta_list: GLib.List
                        ) -> Iterator[pyds.NvDsFrameMeta]:
    """
    Iterate through DeepStream frame metadata GList (doubly linked list).
    """
    # generators catch StopIteration to stop iteration,
    while frame_meta_list is not None:
        yield pyds.glist_get_nvds_frame_meta(frame_meta_list.data)
        # a Glib.List is a doubly linked list where .data is the content
        # and 'next' and 'previous' contain to the next and previous elements
        frame_meta_list = frame_meta_list.next


def obj_meta_iterator(obj_meta_list: GLib.List
                      ) -> Iterator[pyds.NvDsObjectMeta]:
    """
    Iterate through DeepStream object metadata GList (doubly linked list).
    """
    while obj_meta_list is not None:
        yield pyds.glist_get_nvds_object_meta(obj_meta_list.data)
        obj_meta_list = obj_meta_list.next


_ELEM_DOC = """
Create {elem_name} Gst.Element and add to the pipeline.

Returns:
    bool: False on failure, True on success.
"""
"""
Documentation template for a create method
"""


class GstEngine(multiprocessing.Process):
    """
    GstEngine is an internal engine for GStreamer.

    It is a subclass of multiprocessing.Process to run a GLib.MainLoop in
    a separate process. There are several reasons for this:
    
    * GStreamer elements can have memory leaks so if and when the processes
      crashes, it can be restarted without having to restart the whole app.
      In general GStreamer is as buggy as it is fast and the quality of elements
      runs the gamut.
    * Python callbacks called by GLib.MainLoop can block the whole MainLoop.
      (The same is true in C, but you can launch CPU bound stuff in a thread, 
      which is not possible in Python due to the GIL). Running GLib.MainLoop
      it in a separate process and putting the results into a queue if a slot
      is empty (dropping the results if not), avoids this problem.

    Arguments:
        config (:obj:`GstConfig`):
            GstConfig instance for this engine.

    Attributes:
        logger (:obj:`logging.Logger`):
            Python logger for the class.
        queue_timeout (int): 
            (default: 15 seconds) timeout for the queue. If timeout is a
            positive integer, setting/getting properties will block this number
            of seconds maximum before raising the queue.Full exception.

            If queue_timeout is not a positive integer, setting properties
            will put an item into the queue without blocking and if the queue is
            full, a queue.Full exception will be raised.

            General advice: set this to None and catch the exception if
            non-blocking behavior is important (eg. asyncio). Otherwise,
            leave the timeout, but it will block any context you use the
            setter from and this block **cannot** be interrupted, even with
            SIGINT / KeyboardInterrupt.

    Examples:

        NOTE: the default GstConfig pipeline is:
            fakesrc ! identity ... identity ! fakesink,

        >>> source_configs = [dict(),]
        >>> infer_configs = [dict(),]
        >>> config = GstConfig(infer_configs, source_configs)
        >>> engine = GstEngine(config)
        >>> engine.start()
        >>> engine.stop()
        >>> engine.join(10)
        >>> engine.exitcode
        0

        Real-world subclasses can override GstConfig to set different source,
        sink, and inference elements. See GstConfig documentation for details.

    """

    logger = logging.getLogger('GstEngine')
    queue_timeout=10

    def __init__(self, config:GstConfig, *args, **kwargs):
        self.logger.debug('__init__')
        super().__init__(*args, **kwargs)

        # the pipeline configuration
        self._gst_config = config  # type: GstConfig

        # GStreamer main stuff
        self._main_loop = None # type: GLib.MainLoop
        self._pipeline = None  # type: Gst.Pipeline
        # GStreamer elements (in order of connection)
        self._sources = []  # type: List[Gst.Element]
        self._muxer = None  # type: Gst.Element
        self._muxer_lock = multiprocessing.Lock()
        self._tracker = None  # type: Gst.Element
        self._infer_elements = []  # type: List[Gst.Element]
        self._osd = None  # type: Gst.Element
        self._osd_probe_id = None # type: int
        self._sink = None  # type: Gst.Element

        # process communication primitives
        self._result_queue = multiprocessing.Queue(maxsize=1)
        self._stop_requested = multiprocessing.Event()
        self._reset_requested = multiprocessing.Event()

    @property
    def results(self) -> Optional[Detections]:
        """
        Get results waiting in the queue.

        (may block, depending on self.queue_timeout)

        May return None if no result ready.

        Logs to WARNING level on failure to fetch result.
        """
        if not self._result_queue.empty():
            try:
                return self._result_queue.get(block=False, timeout=self.queue_timeout)
            except queue.Empty:
                self.logger.warning("failed to get results from queue (queue.Empty)")
                return None

    def _update_result_queue(self, results):
        """
        Called internally by the GStreamer process.

        Update results queue. Should probably be called by the subclass
        implemetation of on_buffer().

        Does not block (because this would block the GLib.MainLoop).
        
        Can fail if the queue is full in which case the results will
        be dropped and logged to the WARNING level.

        Returns:
            bool: False on failure, True on success.
        """
        if self._result_queue.empty():
            try:
                self._result_queue.put_nowait(results)
                return True
            except queue.Full:
                # this really should't ever happen
                self.logger.warning("failed to put results in queue (queue.Full)")
                return False

    def _create_pipeline(self) -> bool:
        """
        Attempt to create pipeline bin.

        Returns:
            bool: False on failure, True on success.
        """
        # create the pipeline and check
        self.logger.debug('creating pipeline')
        self._pipeline = Gst.Pipeline()
        if not self._pipeline:
            self.logger.error('could not create Gst.Pipeline element')
            return False
        return True

    # TODO(mdegans): some of these creation methods can probably be combined

    def _create_sources(self) -> bool:
        # create a source and check
        for conf in self._gst_config.src_configs:
            self.logger.debug('creating source')
            src = Gst.ElementFactory.make(self._gst_config.SRC_TYPE)  # type: Gst.Element
            if not src:
                self.logger.error(f'could not create source of type: {self._gst_config.SRC_TYPE}')
                return False

            # set properties on the source
            for k, v in conf.items():
                src.set_property(k, v)

            # add the source to the pipeline and check
            if not self._pipeline.add(src):
                self.logger.error('could not add source to pipeline')
                return False
            
            # append the source to the _sources list
            self._sources.append(src)
        return True
    _create_sources.__doc__ = _ELEM_DOC.format(elem_name='`self.config.SRC_TYPE`')

    def _create_muxer(self) -> bool:
        # creeate the muxer and check
        self.logger.debug('creating stream muxer')
        self._muxer = Gst.ElementFactory.make(self._gst_config.MUXER_TYPE)  # type: Gst.Element
        if not self._muxer:
            self.logger.error(
                f'could not create stream muxer of type: {self._gst_config.MUXER_TYPE}')
            return False
        
        # set properties on the muxer
        if self._gst_config.muxer_config:
            for k, v in self._gst_config.muxer_config.items():
                self._muxer.set_property(k, v)
        
        # add the muxer to the pipeline and check
        if not self._pipeline.add(self._muxer):
            self.logger.error('could not add muxer to pipeline')
            return False
        return True
    _create_muxer.__doc__ = _ELEM_DOC.format(elem_name='`self.config.MUXER_TYPE`')

    def _create_tracker(self) -> bool:
        # creeate the tracker and check
        self.logger.debug('creating tracker')
        self._tracker = Gst.ElementFactory.make(self._gst_config.TRACKER_TYPE)  # type: Gst.Element
        if not self._tracker:
            self.logger.error(
                f'could not create tracker of type: {self._gst_config.TRACKER_TYPE}')
            return False

        # set properties on the tracker
        if self._gst_config.tracker_config:
            for k, v in self._gst_config.tracker_config.items():
                self._tracker.set_property(k, v)

        # add the tracker to the pipeline and check
        if not self._pipeline.add(self._tracker):
            self.logger.error('could not add tracker to pipeline')
            return False
        return True
    _create_tracker.__doc__ = _ELEM_DOC.format(elem_name='`self.config.TRACKER_TYPE`')

    def _create_infer_elements(self) -> bool:
        """
        Create GstConfig.INFER_TYPE elements, add them to the pipeline,
        and append them to self._infer_elements for ease of access / linking.
        
        Returns:
            bool: False on failure, True on success.
        """
        self.logger.debug('creating inference elements')
        for conf in self._gst_config.infer_configs:
            # create and check inference element
            elem = Gst.ElementFactory.make(self._gst_config.INFER_TYPE)  # type: Gst.Element
            if not elem:
                self.logger.error(f"failed to create {self._gst_config.INFER_TYPE} element")
                return False

            # set properties on inference element
            for k, v in conf.items():
                elem.set_property(k, v)

            # add the elements to the pipeline and check
            if not self._pipeline.add(elem):
                self.logger.error('could not add source to pipeline')
                return False

            # append the element to the list of inference elements
            self._infer_elements.append(elem)
        return True

    def _create_osd(self) -> bool:
        self.logger.debug('creating osd')
        self._osd = Gst.ElementFactory.make(self._gst_config.OSD_TYPE)
        if not self._osd:
            self.logger.error(f'failed to create {self._gst_config.OSD_TYPE}')
            return False

        # set the osd properties from the config
        if self._gst_config.osd_config:
            for k, v in self._gst_config.osd_config.items():
                self._osd.set_property(k, v)
        
        # add the osd to the pipeline and check
        if not self._pipeline.add(self._osd):
            self.logger.error(f'could not add {self._gst_config.OSD_TYPE} to pipeline')
            return False
        return True

    _create_osd.__doc__ = _ELEM_DOC.format(elem_name='`self.config.OSD_TYPE`')

    def _create_sink(self) -> bool:
        self.logger.debug('creating sink')
        # create the sink element
        self._sink = Gst.ElementFactory.make(self._gst_config.SINK_TYPE)
        if not self._sink:
            self.logger.error(f'failed to create {self._gst_config.SINK_TYPE}')
            return False
        
        # set the sink properties from the config
        if self._gst_config.sink_config:
            for k, v in self._gst_config.sink_config.items():
                self._sink.set_property(k, v)
        
        # add the sink to the pipeline and check
        if not self._pipeline.add(self._sink):
            self.logger.error('could not add sink to pipeline')
            return False
        return True
    _create_sink.__doc__ = _ELEM_DOC.format(elem_name='`self.config.SINK_TYPE`')

    def _create_all(self) -> int:
        """
        Create and link the pipeline from self.config.

        Returns:
            bool: False on failure, True on success.
        """
        create_funcs = (
            self._create_pipeline,
            self._create_sources,
            self._create_muxer,
            self._create_tracker,
            self._create_infer_elements,
            self._create_osd,
            self._create_sink,
        )

        for i, f in enumerate(create_funcs):
            if not f():
                self.logger.error(
                    f"Failed to create DsEngine pipeline at step {i}")
                return False
        return True

    def _on_source_src_pad_create(self, element: Gst.Element, src_pad: Gst.Pad):
        """
        Callback to link sources to the muxer.
        """
        # a lock is required so that identical pads are not requested.
        self._muxer_lock.acquire()
        try:
            # TODO(mdegans): i half expect this to fail since it was broken on Ds4
            muxer_sink = self._muxer.get_request_pad('sink_%u')
            if not muxer_sink:
                self.logger.error(
                    f"failed to get request pad from {self._muxer.name}")
            ret = src_pad.link(muxer_sink)
            if not ret == Gst.PadLinkReturn.OK:
                self.logger.error(
                    f"failed to link source to muxer becase {ret.value_name}")
        finally:
            self._muxer_lock.release()

    def _link_pipeline(self) -> bool:
        """
        Attempt to create infer elements.
        
        Returns:
            bool: False on failure, True on success.
        """
        self.logger.debug('linking pipeline')

        # link the muxer to the tracker
        if not self._muxer.link(self._tracker):
            self.logger.error(
                f'could not link {self._gst_config.MUXER_TYPE} to {self._gst_config.TRACKER_TYPE}')
        
        # arrange for the sources to link to the muxer when they are ready
        for source in self._sources:  # type: Gst.Element
            source.connect('pad-added', self._on_source_src_pad_create)

        # link the tracker to the first inference element
        if not self._tracker.link(self._infer_elements[0]):
            self.logger.error(
                f'could not link {self._gst_config.TRACKER_TYPE} to {self._gst_config.INFER_TYPE}')

        # link the inference elements together
        if not link_many(self._infer_elements):
            return False
        
        # link the final inference element to the sink
        if not self._infer_elements[-1].link(self._osd):
            self.logger.error(
                f'could not link final {self._gst_config.INFER_TYPE} to {self._gst_config.OSD_TYPE}')

        if not self._osd.link(self._sink):
            self.logger.error(
                f'could not link {self._gst_config.OSD_TYPE} to {self._gst_config.SINK_TYPE}')

        self.logger.debug('linking pipeline successful')
        return True

    def on_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo, _: None, ) -> Gst.PadProbeReturn:
        """
        Default source pad probe buffer callback for the sink.
        
        Simply returns Gst.PadProbeReturn.OK, signaling the buffer
        shuould continue down the pipeline.
        """
        return Gst.PadProbeReturn.OK
                
    def stop(self):
        """Stop the GstEngine process."""
        self.logger.info('requesting stop')
        self._stop_requested.set()

    def _quit(self) -> Gst.StateChangeReturn:
        """
        Quit the GLib.MainLoop and set the pipeline to NULL.
        
        Called by _on_stop. A separate function for testing purposes.
        """
        self._main_loop.quit()
        self.logger.debug('shifting pipeline to NULL state')
        ret = self._pipeline.set_state(Gst.State.NULL)
        if ret == Gst.StateChangeReturn.ASYNC:
            ret = self._pipeline.get_state(10)
        if ret != Gst.StateChangeReturn.SUCCESS:
            self.logger.error(
                'Failed to quit cleanly. Self terminating.')
            self.terminate()  # send SIGINT to self

    def _on_stop(self):
        """
        Callback to shut down the process if stop() has been called.
        """
        if self._stop_requested.is_set():
            self.logger.info(f'stopping {self.__class__.__name__}')
            self._quit()
            # clear stop_requested state
            self._stop_requested.clear()
            self.logger.info(f'{self.__class__.__name__} cleanly stopped')

    def run(self):
        """Called on start(). Do not call this directly."""
        self.logger.debug('run() called. Initializing Gstreamer.')
        
        # initialize and check GStreamer
        Gst.init_check()

        # create pipeline,
        # create and add all elements:
        if not self._create_all():
            self.logger.debug('could not create pipeline')
            return -1
        
        # link all pipeline elements:
        if not self._link_pipeline():
            self.logger.error('could not link pipeline')
            return -2
        
        # register pad probe buffer callback on the osd
        self.logger.debug('registering self.on_buffer() callback on osd sink pad')
        osd_sink_pad = self._osd.get_static_pad('sink')
        if not osd_sink_pad:
            self.logger.error('could not get osd sink pad')
            return -3
        self._osd_probe_id = osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self.on_buffer, None)

        # register callback to check for the stop event when idle.
        # TODO(mdegans): test to see if a higher priority is needed.
        self.logger.debug('registering self._on_stop() idle callback with GLib MainLoop')
        GLib.idle_add(self._on_stop)

        # set the pipeline to the playing state
        self.logger.debug('setting pipeline to PLAYING state')
        self._pipeline.set_state(Gst.State.PLAYING)

        # run the main loop.
        # this has a built-in signal handler for SIGINT
        self.logger.debug('creating the GLib.MainLoop')
        self._main_loop = GLib.MainLoop()
        self.logger.debug('starting the GLib.MainLoop')
        self._main_loop.run()

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
