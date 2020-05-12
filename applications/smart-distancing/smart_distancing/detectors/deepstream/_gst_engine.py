import functools
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
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
)

from smart_distancing.detectors.deepstream._ds_config import GstConfig
from smart_distancing import Detections

__all__ = [
    'GstEngine',
    'link_many',
    'PadProbeCallback',
]

PadProbeCallback = Callable[
    [Gst.Pad, Gst.PadProbeInfo, Any],
    Gst.PadProbeReturn,
]
"""
Signature of Gsteamer Pad Probe Callback
"""

# a documentation template for an elemetn creation function
# TODO(mdegans): remove after refactoring elem creation methods
_ELEM_DOC = """
Create {elem_name} Gst.Element and add to the pipeline.

Returns:
    bool: False on failure, True on success.
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
    * Ease of adding and removing new sources. With DeepStream, right now, the
      *easiest* and most reliable way to do this is to relaunch it's process
      with a modified configuration.

    Arguments:
        config (:obj:`GstConfig`):
            GstConfig instance for this engine.
        debug (bool, optional):
            log all bus messages to the debug level
            (this can mean a lot of spam)

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
        IGNORED_MESSAGES(:obj:`tuple` of :obj:`Gst.MessageType`):
            Gst.MessageType to be ignored by on_bus_message.


    Examples:

        NOTE: the default GstConfig pipeline is:
              uridecodebin ! concat ! identity ... identity ! fakesink,

        Real-world subclasses can override GstConfig to set different source,
        sink, and inference elements. See GstConfig documentation for details.

    """

    IGNORED_MESSAGES = tuple()  # type: Tuple[Gst.MessageType]

    logger = logging.getLogger('GstEngine')
    queue_timeout=10

    def __init__(self, config:GstConfig, *args, debug=False, **kwargs):
        self.logger.debug('__init__')
        super().__init__(*args, **kwargs)
        # set debug for optional extra logging
        self._debug = debug

        # the pipeline configuration
        self._gst_config = config  # type: GstConfig

        # GStreamer main stuff
        self._main_loop = None # type: GLib.MainLoop
        self._pipeline = None  # type: Gst.Pipeline
        # GStreamer elements (in order of connection)
        self._sources = []  # type: List[Gst.Element]
        self._muxer = None  # type: Gst.Element
        self._muxer_lock = GLib.Mutex()
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

    def on_bus_message(self, bus: Gst.Bus, message: Gst.Message, *_) -> bool:
        """
        Default bus message callback.

        This implementation does the following on each message type:

        Ignored:
            any Gst.MessageType in GstEngine.IGNORED_MESSAGES
        
        Logged:
            Gst.MessageType.STREAM_STATUS
            Gst.MessageType.STATE_CHANGED
            Gst.MessageType.WARNING
            (all others)

        call self._quit():
            Gst.MessageType.EOS
            Gst.MessageType.ERROR
        """
        # TAG and DURATION_CHANGED seem to be the most common
        if message.type in self.IGNORED_MESSAGES:
            pass
        elif message.type == Gst.MessageType.STREAM_STATUS:
            status, owner = message.parse_stream_status()  # type: Gst.StreamStatusType, Gst.Element
            self.logger.debug(f"{owner.name}:status:{status.value_name}")
        elif message.type == Gst.MessageType.STATE_CHANGED:
            old, new, _ = message.parse_state_changed()  # type: Gst.State, Gst.State, Gst.State
            self.logger.debug(
                f"{message.src.name}:state-change:"
                f"{old.value_name}->{new.value_name}")
        elif message.type == Gst.MessageType.EOS:
            self.logger.debug(f"Got EOS")
            self._quit()
        elif message.type == Gst.MessageType.ERROR:
            err, errmsg = message.parse_error()  # type: GLib.Error, str
            self.logger.error(f'{err}: {errmsg}')
            self._quit()
        elif message.type == Gst.MessageType.WARNING:
            err, errmsg = message.parse_warning()  # type: GLib.Error, str
            self.logger.warning(f'{err}: {errmsg}')
        else:
            if self._debug:
                self.logger.debug(
                    f"{message.src.name}:{Gst.MessageType.get_name(message.type)}")
        return True

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
            self.logger.debug(f'creating source: {self._gst_config.SRC_TYPE}')
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

    def _create_element(self, e_type:str) -> Optional[Gst.Element]:
        """
        Create a Gst.Element and add to the pipeline.

        Arguments:
            e_type (str):
                The FOO_TYPE of elememt to add defined on the config class
                as an attribute eg. MUXER_TYPE, SRC_TYPE... This argument is
                case insensitive. choices are: ('muxer', 'src', 'sink')

                Once the element of the corresponding type on the config is
                made using Gst.ElementFactory.make, it will be added to 
                self._pipeline and assigned to self._e_type. 

        Returns:
            A Gst.Element if sucessful, otherwise None.
        
        Raises:
            AttributeError if e_type doesn't exist on the config and the class.
        """
        # NOTE(mdegans): "type" and "name" are confusing variable names considering
        #  GStreamer's and Python's usage of them. Synonyms anybody?
        e_type = e_type.lower()
        e_name = getattr(self._gst_config, f'{e_type.upper()}_TYPE')
        props = getattr(self._gst_config, f'{e_type}_config')  # type: dict
        self.logger.debug(f'creating {e_type}: {e_name} with props: {props}')

        # make an self.gst_config.E_TYPE_TYPE element
        elem = Gst.ElementFactory.make(e_name)
        if not elem:
            self.logger.error(f'could not create {e_type}: {e_name}')
            return

        # set properties on the element
        if props:
            for k, v in props.items():
                elem.set_property(k, v)
        
        # assign the element to self._e_type
        setattr(self, f'_{e_type}', elem)

        # add the element to the pipeline and check
        if not self._pipeline.add(elem):
            self.logger.error(f'could not add {e_type}: {e_name} to pipeline.')
            return
        return elem

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

    def _create_all(self) -> int:
        """
        Create and link the pipeline from self.config.

        Returns:
            bool: False on failure, True on success.
        """
        create_funcs = (
            self._create_pipeline,
            self._create_sources,
            functools.partial(self._create_element, 'muxer'),
            functools.partial(self._create_element, 'tracker'),
            self._create_infer_elements,
            functools.partial(self._create_element, 'osd'),
            functools.partial(self._create_element, 'sink'),
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
        # GLib.Mutex is required because python's isn't respected by GLib's MainLoop
        self._muxer_lock.lock()
        try:
            self.logger.debug(f'{element.name} new pad: {src_pad.name}')
            self.logger.debug(
                f'{src_pad.name} caps:{src_pad.props.caps}')
            # TODO(mdegans): i half expect this to fail since it was broken on Ds4
            muxer_sink_pad_name = f'sink_{self._muxer.numsinkpads}'
            self.logger.debug(f'{self._muxer.name}:requesting pad:{muxer_sink_pad_name}')
            muxer_sink = self._muxer.get_request_pad(muxer_sink_pad_name)
            if not muxer_sink:
                self.logger.error(
                    f"failed to get request pad from {self._muxer.name}")
            self.logger.debug(
                f'{muxer_sink.name} caps:{muxer_sink.props.caps}')
            ret = src_pad.link(muxer_sink)
            if not ret == Gst.PadLinkReturn.OK:
                self.logger.error(
                    f"failed to link source to muxer becase {ret.value_name}")
                self._quit()
        finally:
            self._muxer_lock.unlock()

    def _link_pipeline(self) -> bool:
        """
        Attempt to create infer elements.
        
        Returns:
            bool: False on failure, True on success.
        """
        self.logger.debug('linking pipeline')
       
        # arrange for the sources to link to the muxer when they are ready
        # (uridecodebin has sometimes pads so needs to be linked by callback)
        for source in self._sources:  # type: Gst.Element
            source.connect('pad-added', self._on_source_src_pad_create)

        # link the muxer to the first inference element
        if not self._muxer.link(self._infer_elements[0]):
            self.logger.error(
                f'could not link {self._muxer.name} to {self._infer_elements[0].name}')
            return False

        if not self._infer_elements[0].link(self._tracker):
            self.logger.error(
                f'could not link primary inference engine to tracker')
            return False

        # if there are secondary inference elements
        if self._infer_elements[1:]:
            # link tracker to the rest of the inference elements
            if not link_many((self._tracker, *self._infer_elements[1:])):
                return False

            # link the final inference element to the osd converter
            if not self._infer_elements[-1].link(self._osd_converter):
                self.logger.error(
                    f'could not link final inference element to {self._osd_converter.name}')
        else:
            # link tracker directly to the osd converter
            if not self._tracker.link(self._osd_converter):
                self.logger.error(
                    f'could not link {self._tracker.name} to {self._osd_converter.name}')

        if not self._osd_converter.link(self._osd):
            self.logger.error(
                f'could not link {self._osd_converter.name} to {self._osd.name}')

        if not self._osd.link(self._sink):
            self.logger.error(
                f'could not link {self._osd.name} to {self._sink.name}')
            return False

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

        # register bus message callback
        bus = self._pipeline.get_bus()
        if not bus:
            self.logger.error('could not get bus')
            return -2
        self.logger.debug('registering bus message callback')
        bus.add_watch(GLib.PRIORITY_DEFAULT, self.on_bus_message, None)

        # link all pipeline elements:
        if not self._link_pipeline():
            self.logger.error('could not link pipeline')
            return -3
        
        # register pad probe buffer callback on the osd
        self.logger.debug('registering self.on_buffer() callback on osd sink pad')
        osd_sink_pad = self._osd.get_static_pad('sink')
        if not osd_sink_pad:
            self.logger.error('could not get osd sink pad')
            return -4
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
