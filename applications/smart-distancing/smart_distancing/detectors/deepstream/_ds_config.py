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

import os
import logging

from math import log, ceil

from typing import (
    Any,
    Iterable,
    Mapping,
    Union,
)

__all__ = [
    'DsConfig',
    'ElemConfig',
    'GstConfig',
]

Path = Union[str, os.PathLike]
ElemConfig = Mapping[str, Any]

class GstConfig(object):
    """
    GstConfig is a simple class to store configuration
    for a GstEngine.

    Arguments:
        infer_configs:
            For each :obj:`ElemConfig` (dict) in this iterable,
            GstEngine will create a new `GstConfig.INFER_TYPE`
            Gst.Element and assign it these properties.
        src_configs:
            For each :obj:`ElemConfig` (dict) in this iterable,
            GstEngine will create a new `GstConfig.SRC_TYPE`
            Gst.Element and assign it these properties.
        muxer_config:
            :obj:`ElemConfig` (a dict) to use to apply properties to
            the muxer of type `GstConfig.MUXER_TYPE`.
        tracker_config:
            :obj:`ElemConfig` (a dict) to use to apply properties to
            the muxer of type `GstConfig.TRACKER_TYPE`.
        osd_config:
            :obj:`ElemConfig` (a dict) to use to apply properties to
            the muxer of type `GstConfig.OSD_TYPE`.
        sink_config:
            :obj:`ElemConfig` (a dict) to use to apply properties to
            the muxer of type `GstConfig.SINK_TYPE`.

    Examples:

        A supplied inference configuration and source configuration are
        the only two required parameters and available after __init__ and
        validate() on their correspondingly named attributes.

        >>> infer_configs = [
        ...     {'uff-file': '/path/to/detector.uff', 'network-mode': 0},
        ...     {'onnx-file': '/path/to/classifier.onnx', 'classifier-async-mode': True},
        ... ]
        >>> src_configs = [
        ...     {'uri', 'https://foo.com/video.mp4'},
        ...     {'uri', 'file:///home/foo/Videos/video.mp4'},
        ... ]
        >>> config = GstConfig(infer_configs, src_configs)
        >>> config.infer_configs == infer_configs
        True
        >>> config.infer_configs[1]['classifier-async-mode']
        True
        >>> config.src_configs == src_configs
        True
    
        **IMPORTANT NOTE: this class does not currently ensure all properties on
        a config exist on a given GStreamer element. TypeError will be raised by
        the GstProcess if an attempt is made to set a property that does not exist.
        This cannot be caught with try, but you can check the GstProcess return code.**

    TODO(mdegans): more examples and doctests
    """

    SRC_TYPE = 'fakesrc'
    SINK_TYPE = 'fakesink'
    MUXER_TYPE = 'identity'
    INFER_TYPE = 'identity'
    OSD_TYPE = 'identity'
    TRACKER_TYPE = 'identity'

    def __init__(self, infer_configs: Iterable[ElemConfig],
                 src_configs: Iterable[ElemConfig],
                 muxer_config: ElemConfig = None,
                 tracker_config: ElemConfig = None,
                 osd_config: ElemConfig = None,
                 sink_config: ElemConfig = None,):
        self.infer_configs = list(infer_configs)
        self.src_configs = list(src_configs)
        self.muxer_config = muxer_config
        self.tracker_config = tracker_config
        self.osd_config = osd_config
        self.sink_config = sink_config
        self.validate()
    
    def validate(self):
        """
        Validate `self`. Called by __init__.

        Checks:
            * there is at least one source
            * there is at least one inference element

        Raises:
            ValueError: if `self` is invalid.
        
        Examples:

            If an empty source is supplied, ValueError is raised:

            >>> empty_iterable = tuple()
            >>> src_configs = [{'prop': 'val'},]
            >>> config = GstConfig(empty_iterable, src_configs)
            Traceback (most recent call last):
                ...
            ValueError: at least one inference config is required
        """
        if not self.infer_configs:
            raise ValueError(
                "at least one inference config is required")
        if not self.src_configs:
            raise ValueError(
                "at least one source config is required")


class DsConfig(GstConfig):
    """
    DeepStream implementation of GstConfig.
    Batch size properties 

    Arguments:
        max_batch_size (int):
            The maximum allowed batch size parameter.
            Defaults to 32, but this should probably be
            lower on platforms like Jetson Nano for best
            performance.
    """
    SRC_TYPE = 'uridecodebin'
    SINK_TYPE = 'hlssink'
    MUXER_TYPE = 'nvstreammux'
    INFER_TYPE = 'nvinfer'
    OSD_TYPE = 'nvdsosd'
    TRACKER_TYPE = 'nvtracker'

    def __init__(self, *args, max_batch_size=32, **kwargs):
        self.max_batch_size = max_batch_size
        super().__init__(*args, **kwargs)

    # TODO(mdegans): validate method
    def validate(self):
        """
        Checks:
            * superclass validators
            * Set optimal batch-size property by rounding up to the next
              power of two with a maximum of self.max_batch_size

            Example:
                >>> infer_configs = [
                ...     {'uff-file': '/path/to/detector.uff', 'network-mode': 0},
                ...     {'onnx-file': '/path/to/classifier.onnx', 'classifier-async-mode': True},
                ... ]
                >>> src_configs = [
                ...     {'uri', 'https://foo.com/video.mp4'},
                ...     {'uri', 'file:///home/foo/Videos/video.mp4'},
                ...     {'url', 'https://bar.com/video.mp4'},
                ... ]
                >>> config = DsConfig(infer_configs, src_configs)
                >>> config.infer_configs[0]['batch-size']
                4
                >>> config.muxer_config['batch-size']
                4
        """
        super().validate()

        # set the optimal batch size (next power of 2)
        # the formula has mixed reviews on stackoverflow and
        # might fail in some languages, but not in Python.
        num_sources = len(self.src_configs)
        optimal_batch_size = pow(2, ceil(log(num_sources)/log(2)))
        configs_with_batch_size = [
            *self.infer_configs,
        ]
        if not self.muxer_config:
            self.muxer_config = dict()
            configs_with_batch_size.append(self.muxer_config)
        for c in configs_with_batch_size:
            c['batch-size'] = min(optimal_batch_size, self.max_batch_size)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
