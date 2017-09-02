''' Flags for the model '''
import tensorflow as tf

def load_flags(mode = 'train'):
    flags = tf.app.flags
    flags.DEFINE_integer("nfilt", 40, "Number of filters in mel specta")
    flags.DEFINE_float("keep-prob", 0.8, "Keep probability for dropout")
    flags.DEFINE_integer("max-output-len", 200, "Maximum output length for Beam Search")
    flags.DEFINE_integer("rnn-size", 256, "Size of LSTM")
    flags.DEFINE_integer("num-layers", 3, "Number of LSTM Layers for encoding layer")
    flags.DEFINE_integer("num-decoding-layers", 3, "Number of LSTM Layers for decoding layer")
    flags.DEFINE_integer("batch-size", 24, "Batch size of data")
    flags.DEFINE_float("learning-rate", 0.001, "Learning rate")
    flags.DEFINE_integer("num-epochs", 16, "Number of epochs")
    flags.DEFINE_integer("beam-width", 8, "Beam width. Must be lesser than vocab size")
    flags.DEFINE_float("learning-rate-decay", 0.9998, "Learning rate decay")
    flags.DEFINE_float("min-learning-rate", 0.0002, "Minimum learning rate")
    flags.DEFINE_integer("display-step", 20, "Check training loss after every display-step batches")
    flags.DEFINE_string("data-dir", None, "Directory of data")
    flags.DEFINE_string("job-dir", './job/', "Directory in which summary and checkpoint is stored")
    flags.DEFINE_string("checkpoint-path", None, "Load a trained model")
    flags.DEFINE_integer("best-n-inference", 1, "Take best of n for beam search")
    flags.DEFINE_integer("eval-only", False, "Only evaluate. --checkpoint-path is required")
    flags.DEFINE_string("fst-path", None, "Path of language FST")
    flags.DEFINE_integer("cutoff-range", 100, "Attention cutoff from previous mean range")
    flags.DEFINE_bool("use-inference-lm", False, "Use LM during inference")
    flags.DEFINE_bool("use-train-lm", False, "Use LM during training")
    flags.DEFINE_bool("use-conv-feat-att", False, "Use convolutional features for attention")
    flags.DEFINE_integer("num-conv-layers", 2,
        "Number of convolutional layer to use before BLSTM layers")
    flags.DEFINE_integer("conv-layer-width", 5, "Width of convoulutional layer")
    flags.DEFINE_integer("conv-layer-size", 128, "Number of filters")
    if mode == 'train':
        flags.DEFINE_string("trans-file", None, "Path of transcription file")
    if mode == 'infer':
        flags.DEFINE_string("audio-file", None, "Path of audio file for inference")
