"""Log handler for Json Lines format."""
import os
import json
import logging.handlers

__all__ = ['JsonLinesLogger']

class JsonLinesLogger(logging.handlers.TimedRotatingFileHandler):
    """
    Json lines (.jl .jsonl) format TimedRotatingFileHandler.

    See logging.handlers.TimedRotatingFileHandler for full documentation.

    Example usage/test:
    >>> filename = '/tmp/test.log.jl'
    >>> logger = logging.getLogger(__name__)
    >>> jh = JsonLinesLogger(filename)
    >>> logger.addHandler(jh)
    >>> logger.error("test")
    >>> with open(filename) as f:
    ...     for line in f:
    ...         rec = json.loads(line)
    ...         assert rec['level'] == 'ERROR'
    ...         assert rec['msg'] == 'test'
    """
    # TODO(mdegans): change the above test to use a random filename
    #  right now it's too brittle if run multiple times

    def emit(self, record: logging.LogRecord) -> None:
        json_record = {
            'level': record.levelname,
            'time': int(record.created),
            'msg': record.msg,
        }
        # noinspection PyUnresolvedReferences
        self.stream.write(json.dumps(json_record) + os.linesep)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
