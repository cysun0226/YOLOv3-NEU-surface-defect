import tensorflow as tf
import logging
import datetime
import os

FORMAT = '%(message)s'
LOG_LOCATION = './log'
LOG_FILE_FORMAT = '%Y-%m-%d_%H:%M:%S.log'
DATE_FORMAT = '%H:%M:%S'

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        self.writer.add_summary(summary, step)

def set_logging(log_file_name):
    logging.basicConfig(level=logging.INFO, filename=os.path.join(LOG_LOCATION, log_file_name+'.txt'), 
                        filemode='w', format=FORMAT, datefmt=DATE_FORMAT)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
    # also print to stdout
    logging.getLogger().addHandler(stdout_handler)