#!/usr/bin/python3
import os
import logging

import smart_distancing as sd

from typing import (
    Union,
)

logger = logging.getLogger(__name__)

__all__ = ['main']

def load_config(config: str) -> sd.core.ConfigEngine:
    """
    :return: a ConfigEngine for an absolute or relativefilename.\
        
    Searches config dirs in order specified by sd.DEFAULT_TO_USER_PATHS
    """
    logger.debug(f"load_config:config selected: {config}")
    # if it exists as a filename, load it
    if os.path.isfile(config):
        return sd.core.ConfigEngine(config)
    
    # if it exists in the config paths, try loading it
    if config in sd.utils.file.config_files():
        try:
            logger.debug(f"trying {sd.CONFIG_DIR}")
            return sd.core.ConfigEngine(
                os.path.join(sd.CONFIG_DIR, config))
        except FileNotFoundError:
            logger.debug(f"trying: {sd.FALLBACK_CONFIG_DIR}")
            return sd.core.ConfigEngine(
                os.path.join(sd.FALLBACK_CONFIG_DIR, config))


def main(config: Union[str, sd.core.ConfigEngine]) -> int:
    """
    Main entrypoint for smart_distancing.

    :param config: path to a config file or a ConfigEngine instance.
    :return: an integer return code for sys.exit()
    """
    logger.debug(f'main("{config}")')
    if not isinstance(config, sd.core.ConfigEngine):
        config = load_config(config)
    app = sd.core.DefaultDistancing(config)
    app.ui.start()
    return 0


def cli_main() -> int:
    """
    Command line entrypoint for smart_distancing.

    Does the standard argparse dance and sets up logging.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
        choices=sd.utils.file.config_files(),
        required=True,)
    parser.add_argument('--verbose', action='store_true')

    # parse arguments
    args = parser.parse_args()

    # load the config from file
    config = load_config(args.config)

    # override the handler class if set
    handler_cls = sd.loggers.CsvHandler
    try:
        logger_name = config.get_section_dict('Logger')['Name']
        if logger_name in ('json', 'jl'):
            handler_cls = sd.loggers.JsonLinesHandler
    except KeyError:
        pass

    # try to get the log path
    logger_path = sd.LOG_DIR
    try:
        logger_path = config.get_section_dict('Logger')['LogDirectory']
    except KeyError:
        pass   

    # instantiate the handler
    file_handler = handler_cls(
        os.path.join(logger_path, 'smart_distancing' + handler_cls.EXT))
    stream_handler = logging.StreamHandler()
    file_handler.addFilter(sd.loggers.spam_filter)
    stream_handler.addFilter(sd.loggers.spam_filter)

    # tell logging to use the appropriate level and stream handler
    logging.basicConfig(
        handlers=[file_handler, stream_handler],
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    del args.verbose  # we don't need this anymore, and it'll break below

    # convert the args to a dict and unpack it
    # allowing easier reuse of main without argparse
    # it also has the side benefit of enforcing:
    # argparse arguments == main arguments
    return main(**vars(args))


if __name__ == '__main__':
    import sys
    sys.exit(cli_main())
