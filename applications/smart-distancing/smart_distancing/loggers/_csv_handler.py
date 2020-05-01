"""Log handler for .csv format."""
import logging.handlers
import csv

__all__ = ['CsvHandler']

class CsvHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Csv format TimedRotatingFileHandler.

    See logging.handlers.TimedRotatingFileHandler for full documentation.

    Example usage/test:

    >>> filename = '/tmp/test.log.csv'
    >>> logger = logging.getLogger(__name__)
    >>> ch = CsvHandler(filename)
    >>> logger.addHandler(ch)
    >>> logger.error("test")
    >>> ch.stream.close()
    >>> with open(filename) as f:
    ...     for line in f:
    ...         assert 'ERROR' in line
    ...         assert 'test' in line
    """
    # TODO(mdegans): change the above test to use a random filename
    #  right now it's too brittle if run multiple times

    FIELD_NAMES = ('time', 'level', 'msg')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO(mdegans): test rotation works, since this might break
        #  and keep writing to the initial fd. May need to override something
        #  else so that .csv_writer is recreated when .stream is,
        #  or pass csv.DictWriter a property returning the stream.
        self.csv_writer = csv.DictWriter(
            self.stream, fieldnames=self.FIELD_NAMES)

    def emit(self, record: logging.LogRecord) -> None:
        csv_record = {
            'level': record.levelname,
            'time': int(record.created),
            'msg': record.msg,
        }
        # noinspection PyUnresolvedReferences
        self.csv_writer.writerow(csv_record)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
