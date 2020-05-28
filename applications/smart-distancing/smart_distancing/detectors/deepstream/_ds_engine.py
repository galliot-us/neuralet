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
import configparser
import tempfile
import logging
import queue

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
# import metadata stuff
from smart_distancing.distance_pb2 import (
    Batch,
    Frame,
    Person,
    BBox,
)
from smart_distancing.detectors.deepstream._gst_engine import GstEngine
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
    'DsEngine',
    'frame_meta_iterator',
    'obj_meta_iterator',
]

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


def write_config(tmpdir, config:dict) -> str:
    """
    Write a nvinfer config to a .ini file in tmpdir and return the filename.

    The section heading is [property]

    Example:
        >>> config = {
        ...     'model-file': 'foo.caffemodel',
        ...     'proto-file': 'foo.prototxt',
        ...     'labelfile-path': 'foo.labels.txt',
        ...     'int8-calib-file': 'foo_cal_trt.bin',
        ... }
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     filename = write_config(tmp, config)
        ...     print(filename)
        ...     with open(filename) as f:
        ...         for l in f:
        ...             print(l, end='')
        /tmp/tmp.../config....ini
        [property]
        model-file = foo.caffemodel
        proto-file = foo.prototxt
        labelfile-path = foo.labels.txt
        int8-calib-file = foo_cal_trt.bin
        <BLANKLINE>
    """
    # TODO(mdegans): simple property validation to fail fast
    cp = configparser.ConfigParser()
    cp['property'] = config
    fd, filename = tempfile.mkstemp(prefix='config', suffix='.ini', dir=tmpdir, text=True)
    with open(fd, 'w') as f:
        cp.write(f)
    return filename

class DsEngine(GstEngine):
    """
    DeepStream implemetation of GstEngine.
    """

    _tmp = None  # type: tempfile.TemporaryDirectory
    _previous_scores = None

    def _quit(self):
        # cleanup the temporary directory we created on __init__
        self._tmp.cleanup()
        # this can self terminate so it should be called last:
        super()._quit()

    @property
    def tmp(self):
        """
        Path to the /tmp/ds_engine... folder used by this engine.

        This path is normally deleted on self._quit()
        """
        return self._tmp.name

    def on_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo, _: None, ) -> Gst.PadProbeReturn:
        """
        Parse inference metadata and put it in the result queue.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error(
                'Failed to get buffer from Gst.PadProbeInfo. Removing probe.')
            return Gst.PadProbeReturn.REMOVE
        
        # the __hash__ of a gst_buffer is a pointer
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        frames = []

        # iterate through the metadata and add results to the list
        for frame_meta in frame_meta_iterator(batch_meta.frame_meta_list):
            people = []
            for obj_meta in obj_meta_iterator(frame_meta.obj_meta_list):
                # we need an object id assigned by the tracker to know what
                # box to color when the results come back.  likewise we can
                # only score objects whose uid still exists when the scores
                # come back in the self.osd_queue
                if obj_meta.object_id:
                    people.append(Person(
                        uid=obj_meta.object_id,
                        bbox=BBox(
                            left=int(obj_meta.rect_params.left),
                            top=int(obj_meta.rect_params.top),
                            width=int(obj_meta.rect_params.width),
                            height=int(obj_meta.rect_params.height),
                        )
                    ))
                else:
                    # todo(mdegans): do some default/test drawing here?
                    pass
            frames.append(Frame(
                frame_num=frame_meta.frame_num,
                source_id=frame_meta.source_id,
                people=people
            ))

        batch = Batch(frames=frames)
        batch_str = batch.SerializeToString()

        # we try to update the results queue, but it might be full if
        # the results queue is full becauase the ui process is too slow
        if not self._update_result_queue(batch_str):
            # NOTE(mdegans): we can drop the whole buffer here if we want to drop
            # frames when we're unable to update the metadata queue
            # return Gst.PadProbeReturn.DROP
            pass

        # return pad probe ok, which passes the buffer on
        return Gst.PadProbeReturn.OK

    def run(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='ds_engine')
        super().run()

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
