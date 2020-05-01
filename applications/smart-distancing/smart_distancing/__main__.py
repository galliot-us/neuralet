#!/usr/bin/python3
import smart_distancing as sd

from typing import (
    Union,
)

__all__ = ['main']

def main(config: Union[str, sd.core.ConfigEngine]) -> int:
    """
    Main entrypoint for smart_distancing.

    :param config: path to a config file or a ConfigEngine instance.
    :return: an integer return code for sys.exit()
    """
    app = sd.core.DefaultDistancing(config)
    app.ui.start()
    return 0


def cli_main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    # convert the args to a dict and unpack it
    # allowing easier reuse of main without argparse
    # it also has the side benefit of enforcing:
    # argparse arguments == main arguments
    return main(**vars(args))


if __name__ == '__main__':
    import sys
    sys.exit(cli_main())