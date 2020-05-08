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

        # a list to store the detections
        results = []

        # iterate through the metadata and add results to the list
        for frame_meta in frame_meta_iterator(batch_meta.frame_meta_list):
            for obj_meta in obj_meta_iterator(frame_meta.obj_meta_list):
                rect = obj_meta.rect_params  # type: pyds.NvOSD_RectParams
                results.append({
                    'fnum': frame_meta.frame_num,
                    'id': obj_meta.class_id,
                    'bbox': [
                        rect.left,  # x1
                        rect.top,  # y1
                        rect.left + rect.width,  # x2
                        rect.top + rect.height,  # y2
                    ],
                    'score': obj_meta.confidence,
                })

        if not self._update_result_queue(results):
            # NOTE(mdegans): we can drop the whole buffer here if we want to drop
            # frames when we're unable to update the metadata queue
            # return Gst.PadProbeReturn.DROP
            pass

        # return pad probe ok, which passes the buffer on
        return Gst.PadProbeReturn.OK

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
            self.logger.debug(f'writing config: {conf}')
            elem.set_property('config-file-path', write_config(self.tmp, conf))

            # add the elements to the pipeline and check
            if not self._pipeline.add(elem):
                self.logger.error('could not add source to pipeline')
                return False

            # append the element to the list of inference elements
            self._infer_elements.append(elem)
        return True

    def run(self):
        self._tmp = tempfile.TemporaryDirectory(prefix='ds_engine')
        super().run()

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
