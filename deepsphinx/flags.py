''' Flags for the model '''
import tensorflow as tf

#https://github.com/tensorflow/tensorflow/blob/a1fba7f5ac3de39b106af36c3737ea854f09e9ac/tensorflow/python/platform/flags.py
import argparse as _argparse

_global_parser = _argparse.ArgumentParser()

# pylint: disable=invalid-name

class _FlagValues(object):
  """Global container and accessor for flags and their values."""

  def __init__(self):
    self.__dict__['__flags'] = {}
    self.__dict__['__parsed'] = False
    self.__dict__['__required_flags'] = set()

  def _parse_flags(self, args=None):
    result, unparsed = _global_parser.parse_known_args(args=args)
    for flag_name, val in vars(result).items():
      self.__dict__['__flags'][flag_name] = val
    self.__dict__['__parsed'] = True
    self._assert_all_required()
    return unparsed

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the flag --name."""
    try:
      parsed = self.__dict__['__parsed']
    except KeyError:
      # May happen during pickle.load or copy.copy
      raise AttributeError(name)
    if not parsed:
      self._parse_flags()
    if name not in self.__dict__['__flags']:
      raise AttributeError(name)
    return self.__dict__['__flags'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    self.__dict__['__flags'][name] = value
    self._assert_required(name)

  def _add_required_flag(self, item):
    self.__dict__['__required_flags'].add(item)

  def _assert_required(self, flag_name):
    if (flag_name not in self.__dict__['__flags'] or
        self.__dict__['__flags'][flag_name] is None):
      raise AttributeError('Flag --%s must be specified.' % flag_name)

  def _assert_all_required(self):
    for flag_name in self.__dict__['__required_flags']:
      self._assert_required(flag_name)


def _define_helper(flag_name, default_value, docstring, flagtype):
  """Registers 'flag_name' with 'default_value' and 'docstring'."""
  _global_parser.add_argument('--' + flag_name,
                              default=default_value,
                              help=docstring,
                              type=flagtype)


# Provides the global object that can be used to access flags.
FLAGS = _FlagValues()


def DEFINE_string(flag_name, default_value, docstring):
  """Defines a flag of type 'string'.
  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a string.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, str)


def DEFINE_integer(flag_name, default_value, docstring):
  """Defines a flag of type 'int'.
  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as an int.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, int)


def DEFINE_boolean(flag_name, default_value, docstring):
  """Defines a flag of type 'boolean'.
  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """
  # Register a custom function for 'bool' so --flag=True works.
  def str2bool(v):
    return v.lower() in ('true', 't', '1')
  _global_parser.add_argument('--' + flag_name,
                              nargs='?',
                              const=True,
                              help=docstring,
                              default=default_value,
                              type=str2bool)

  # Add negated version, stay consistent with argparse with regard to
  # dashes in flag names.
  _global_parser.add_argument('--no' + flag_name,
                              action='store_false',
                              dest=flag_name.replace('-', '_'))


# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean  # pylint: disable=invalid-name


def DEFINE_float(flag_name, default_value, docstring):
  """Defines a flag of type 'float'.
  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a float.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, float)


def mark_flag_as_required(flag_name):
  """Ensures that flag is not None during program execution.
  
  It is recommended to call this method like this:
  
    if __name__ == '__main__':
      tf.flags.mark_flag_as_required('your_flag_name')
      tf.app.run()
  
  Args:
    flag_name: string, name of the flag to mark as required.
 
  Raises:
    AttributeError: if flag_name is not registered as a valid flag name.
      NOTE: The exception raised will change in the future. 
  """
  if _global_parser.get_default(flag_name) is not None:
    print(
        'Flag %s has a non-None default value; therefore, '
        'mark_flag_as_required will pass even if flag is not specified in the '
        'command line!' % flag_name)
  FLAGS._add_required_flag(flag_name)


def mark_flags_as_required(flag_names):
  """Ensures that flags are not None during program execution.
  
  Recommended usage:
  
    if __name__ == '__main__':
      tf.flags.mark_flags_as_required(['flag1', 'flag2', 'flag3'])
      tf.app.run()
  
  Args:
    flag_names: a list/tuple of flag names to mark as required.
  Raises:
    AttributeError: If any of flag name has not already been defined as a flag.
      NOTE: The exception raised will change in the future.
  """
  for flag_name in flag_names:
    mark_flag_as_required(flag_name)


def load_flags(mode = 'train'):
    DEFINE_integer('nfilt', 40, 'Number of filters in mel specta')
    DEFINE_float('keep-prob', 0.75, 'Keep probability for dropout')
    DEFINE_integer('max-output-len', 300, 'Maximum output length for Beam Search')
    DEFINE_integer('rnn-size', 256, 'Size of LSTM')
    DEFINE_integer('num-layers', 4, 'Number of LSTM Layers for encoding layer')
    DEFINE_integer('num-decoding-layers', 1, 'Number of LSTM Layers for decoding layer')
    DEFINE_integer('batch-size', 24, 'Batch size of data')
    DEFINE_float('learning-rate', 0.001, 'Learning rate')
    DEFINE_integer('num-epochs', 16, 'Number of epochs')
    DEFINE_integer('beam-width', 20, 'Beam width. Must be lesser than vocab size')
    DEFINE_float('learning-rate-decay', 1.0, 'Learning rate decay')
    DEFINE_float('min-learning-rate', 0.0002, 'Minimum learning rate')
    DEFINE_integer('display-step', 20, 'Check training loss after every display-step batches')
    DEFINE_string('data-dir', None, 'Directory of data')
    DEFINE_string('job-dir', './job/', 'Directory in which summary and checkpoint is stored')
    DEFINE_string('checkpoint-path', None, 'Load a trained model')
    DEFINE_bool('eval-only', False, 'Only evaluate. --checkpoint-path is required')
    DEFINE_string('fst-path', None, 'Path of language FST')
    DEFINE_integer('cutoff-range', 1000, 'Attention cutoff from previous mean range')
    DEFINE_bool('use-inference-lm', False, 'Use LM during inference')
    DEFINE_bool('use-train-lm', False, 'Use LM during training')
    DEFINE_bool('use-conv-feat-att', True, 'Use convolutional features for attention')
    DEFINE_integer('num-conv-layers', 0,
        'Number of convolutional layer to use before BLSTM layers')
    DEFINE_integer('conv-layer-width', 5, 'Width of convoulutional layer')
    DEFINE_integer('conv-layer-size', 128, 'Number of filters')
    if mode == 'train':
        DEFINE_string('trans-file', None, 'Path of transcription file')
    if mode == 'infer':
        DEFINE_string('audio-file', None, 'Path of audio file for inference')
