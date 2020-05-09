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

from math import (
    log,
    ceil,
    sqrt,
)

from typing import (
    TYPE_CHECKING,
    Any,
    Tuple,
    Iterable,
    Mapping,
    Union,
)
if TYPE_CHECKING:
    from smart_distancing.core._config_engine import ConfigEngine
else:
    ConfigEngine = None

__all__ = [
    'DsConfig',
    'ElemConfig',
    'GstConfig',
]

Path = Union[str, os.PathLike]
ElemConfig = Mapping[str, Any]


def calc_rows_and_columns(num_sources: int) -> int:
    """
    Calculate rows and columns values from a number of sources.

    Returns:
        (int) math.ceil(math.sqrt(num_sources))
    """
    if not num_sources:
        return 1
    return int(ceil(sqrt(num_sources)))


def calc_tile_resolution(out_res: Tuple[int, int], rows_and_columns: int) -> Tuple[int, int]:
    """
    Return the optimal resolution for the stream muxer to scale input sources to.
    (same as the resolution for a tile).
    """
    return out_res[0] // rows_and_columns, out_res[1] // rows_and_columns


class GstConfig(object):
    """
    GstConfig is a simple class to store configuration for a GstEngine.

    Arguments:
        infer_configs:
            For each :obj:`ElemConfig` (dict) in this iterable,
            GstEngine will create a new `GstConfig.INFER_TYPE`
            Gst.Element and assign it these properties.
        src_configs:
            For each :obj:`ElemConfig` (dict) in this iterable,
            GstEngine will create a new `GstConfig.SRC_TYPE`
            Gst.Element and assign it these properties.
            
            NOTE: the default linking implementation on GstEngine
            assumes the src pad type has a `Sometimes` pad
            (eg. uridecodebin)
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

    SRC_TYPE = 'uridecodebin'
    SINK_TYPE = 'fakesink'
    MUXER_TYPE = 'concat'
    INFER_TYPE = 'identity'
    OSD_TYPE = 'identity'
    TRACKER_TYPE = 'identity'

    def __init__(self, master_config: ConfigEngine,
                 infer_configs: Iterable[ElemConfig],
                 src_configs: Iterable[ElemConfig],
                 muxer_config: ElemConfig = None,
                 tracker_config: ElemConfig = None,
                 osd_config: ElemConfig = None,
                 sink_config: ElemConfig = None,):
        self.master_config = master_config
        self.infer_configs = list(infer_configs)
        self.src_configs = list(src_configs)
        self.muxer_config = muxer_config
        self.tracker_config = tracker_config
        self.osd_config = osd_config
        self.sink_config = sink_config
        self.validate()

    @property
    def rows_and_columns(self) -> int:
        """
        Number of rows and columns for the tiler element.
        
        Calculated based on the number of sources.
        """
        return calc_rows_and_columns(len(self.src_configs))

    @property
    def tile_resolution(self) -> Tuple[int, int]:
        """
        Resolution of an individual video tile.
        
        Calculated based on the resolution and number of sources.
        """
        return calc_tile_resolution(self.out_resolution, self.rows_and_columns)

    @property
    def out_resolution(self) -> Tuple[int, int]:
        """
        Output video resolution as a 2 tuple of width, height.

        Read from self.master_config.config['App']
        """
        return tuple(int(i) for i in self.master_config.config['App']['Resolution'].split(','))
    
    @property
    def host(self) -> str:
        """
        Host to serve on.

        Read from self.master_config.config['App']
        """
        return self.master_config.config['App']['Host']

    @property
    def port(self) -> int:
        """
        Port to serve on.
        """
        return int(self.master_config.config['App']['Port'])

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

    @property
    def batch_size(self) -> int:
        """
        Return the optimal batch size.
        (next power of two up from the number of sources).

        TODO(mdegans): it's unclear if this is actually optimal
          and under what circumstances (depends on model, afaik)
          tests must be run to see if it's better to use the number
          of sources directly.

        Control the max by setting max_batch_size.
        """
        optimal = pow(2, ceil(log(len(self.src_configs))/log(2)))
        return min(optimal, self.max_batch_size)

    # TODO(mdegans): split this up into multiple functions
    def validate(self):
        """
        Checks:
            * superclass validators
            * Set optimal batch-size property by rounding up to the next
              power of two with a maximum of self.max_batch_size
            * override muxer resolution to self.tile_resolution
        """
        # TODO(mdegans): a doctest would be too long, so a unit test file
        #  is necessary for this class and method.
        super().validate()

        # create muxer config if it doesn't exist
        if not self.muxer_config:
            self.muxer_config = dict()

        # at a minimum, a muxer config must have a resolution
        self.muxer_config.update({
            'width': self.tile_resolution[0],
            'height': self.tile_resolution[1],
        })

        # override the batch size
        # all have to match or bad things happen
        configs_with_batch_size = [
            *self.infer_configs, self.muxer_config]
        for c in configs_with_batch_size:
            c['batch-size'] = self.batch_size

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
