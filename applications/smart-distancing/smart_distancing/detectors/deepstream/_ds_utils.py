"""
DeepStream common utilities.
"""
# Copyright (c) 2020 Michael de Gans
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os
import shutil
import subprocess

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import (
    Gst,
    GLib,
)

from typing import (
    Tuple,
    Optional,
)

__all__ = [
    'bin_to_pdf',
    'find_deepstream',
]

DS_VERSIONS = ('4.0', '5.0')
DS4_PATH = '/opt/nvidia/deepstream/deepstream-{ver}'

logger = logging.getLogger(__name__)

def find_deepstream() -> Tuple[str, str]:
    """
    Finds DeepStream.

    Return:
        A 2 tuple of the DeepStream version
        and it's root path or None if no
        version is found.
    """
    # TODO(mdegans): implement
    for ver in DS_VERSIONS:
        ds_dir = DS4_PATH.format(ver=ver)
        if os.path.isdir(ds_dir):
            return ver, ds_dir

# this is from `mce.pipeline`
def bin_to_pdf(bin_: Gst.Bin, details: Gst.DebugGraphDetails, filename: str,
               ) -> Optional[str]:
    """
    Copied from `mce.pipeline <https://pypi.org/project/mce/>`_

    Dump a Gst.Bin to pdf using 
    `Gst.debug_bin_to_dot_file <https://lazka.github.io/pgi-docs/Gst-1.0/functions.html#Gst.debug_bin_to_dot_file>`_
    and graphviz.
    Will launch the 'dot' subprocess in the background with Popen.
    Does not check whether the process completes, but a .dot is
    created in any case. Has the same arguments as 
    `Gst.debug_bin_to_dot_file <https://lazka.github.io/pgi-docs/Gst-1.0/functions.html#Gst.debug_bin_to_dot_file>`_

    Arguments:
        bin:
            the bin to make a .pdf visualization of
        details:
            a Gst.DebugGraphDetails choice (see gstreamer docs)
        filename:
            a base filename to use (not full path, with no extension)
            usually this is the name of the bin you can get with some_bin.name

    Returns:
        the path to the created file (.dot or .pdf) or None if
        GST_DEBUG_DUMP_DOT_DIR not found in os.environ
    """
    if 'GST_DEBUG_DUMP_DOT_DIR' in os.environ:
        dot_dir = os.environ['GST_DEBUG_DUMP_DOT_DIR']
        dot_file = os.path.join(dot_dir, f'{filename}.dot')
        pdf_file = os.path.join(dot_dir, f'{filename}.pdf')
        logger.debug(f"writing {bin_.name} to {dot_file}")
        Gst.debug_bin_to_dot_file(bin_, details, filename)
        dot_exe = shutil.which('dot')
        if dot_exe:
            logger.debug(
                f"converting {os.path.basename(dot_file)} to "
                f"{os.path.basename(pdf_file)} in background")
            command = ('nohup', dot_exe, '-Tpdf', dot_file, f'-o{pdf_file}')
            logger.debug(
                f"running: {' '.join(command)}")
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp,
            )
        else:
            logger.warning(
                f'graphviz does not appear to be installed, so cannot convert'
                f'{dot_file} to pdf. You can install graphviz with '
                f'"sudo apt install graphviz" on Linux for Tegra or Ubuntu.')
            return dot_file
        return pdf_file
    return None
