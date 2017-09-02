Training
========

Data preparation
----------------

Data contains transcripts file and audio files. The format of transcription
file is csv in the form of `<transcript>,<set_id>,<speaker_id>,<file path>` for
each transcript. If speaker ID is not available, any word(e.g `None`) can be
used to normalize the features by complete dataset.

To change the vocabulary, you need to edit the python code in `vocab.py`

Usage
-----

Install `tensorflow`, `python_speech_features` and `openfst`(optional) first before
running, and generate FST if set in command line options.


```
usage: deepsphinx-train [-h] [--nfilt NFILT] [--keep-prob KEEP_PROB]
                        [--max-output-len MAX_OUTPUT_LEN]
                        [--rnn-size RNN_SIZE] [--num-layers NUM_LAYERS]
                        [--num-decoding-layers NUM_DECODING_LAYERS]
                        [--batch-size BATCH_SIZE]
                        [--learning-rate LEARNING_RATE]
                        [--num-epochs NUM_EPOCHS] [--beam-width BEAM_WIDTH]
                        [--learning-rate-decay LEARNING_RATE_DECAY]
                        [--min-learning-rate MIN_LEARNING_RATE]
                        [--display-step DISPLAY_STEP] [--data-dir DATA_DIR]
                        [--job-dir JOB_DIR]
                        [--checkpoint-path CHECKPOINT_PATH]
                        [--best-n-inference BEST_N_INFERENCE]
                        [--eval-only EVAL_ONLY] [--fst-path FST_PATH]
                        [--cutoff-range CUTOFF_RANGE]
                        [--use-inference-lm [USE_INFERENCE_LM]]
                        [--nouse-inference-lm] [--use-train-lm [USE_TRAIN_LM]]
                        [--nouse-train-lm]
                        [--use-conv-feat-att [USE_CONV_FEAT_ATT]]
                        [--nouse-conv-feat-att] [--trans-file TRANS_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --nfilt NFILT         Number of filters in mel specta
  --keep-prob KEEP_PROB
                        Keep probability for dropout
  --max-output-len MAX_OUTPUT_LEN
                        Maximum output length for Beam Search
  --rnn-size RNN_SIZE   Size of LSTM
  --num-layers NUM_LAYERS
                        Number of LSTM Layers for encoding layer
  --num-decoding-layers NUM_DECODING_LAYERS
                        Number of LSTM Layers for decoding layer
  --batch-size BATCH_SIZE
                        Batch size of data
  --learning-rate LEARNING_RATE
                        Learning rate
  --num-epochs NUM_EPOCHS
                        Number of epochs
  --beam-width BEAM_WIDTH
                        Beam width. Must be lesser than vocab size
  --learning-rate-decay LEARNING_RATE_DECAY
                        Learning rate decay
  --min-learning-rate MIN_LEARNING_RATE
                        Minimum learning rate
  --display-step DISPLAY_STEP
                        Check training loss after every display-step batches
  --data-dir DATA_DIR   Directory of data
  --job-dir JOB_DIR     Directory in which summary and checkpoint is stored
  --checkpoint-path CHECKPOINT_PATH
                        Load a trained model
  --best-n-inference BEST_N_INFERENCE
                        Take best of n for beam search
  --eval-only EVAL_ONLY
                        Only evaluate. --checkpoint-path is required
  --fst-path FST_PATH   Path of language FST
  --cutoff-range CUTOFF_RANGE
                        Attention cutoff from previous mean range
  --use-inference-lm [USE_INFERENCE_LM]
                        Use LM during inference
  --nouse-inference-lm
  --use-train-lm [USE_TRAIN_LM]
                        Use LM during training
  --nouse-train-lm
  --use-conv-feat-att [USE_CONV_FEAT_ATT]
                        Use convolutional features for attention
  --nouse-conv-feat-att
  --trans-file TRANS_FILE
                        Path of transcription file
```


Language model
--------------

The model also supports using FST based LM. See file `make_fst` for commands
for generating Language model FST.

API Usage
---------

There is a minimal API present for prediction of sequence from audio files. As
it is not fully complete, it is recommended to read it before using it. The
usage is intended to be something like this:

```
from deepsphinx.api import Predict
ds = Predict('path/to/checkpoint').predict('path/to/audio')
```

Inference
=========

There is a tool for inference on single file. We normalize the features by file,
and so the accuracy might be lower as for training we normalize by speaker.

Run this command for inference:

```
deepsphinx-infer --checkpoint-path <ckpt> --audio-file <audio-file>
```

To get all the options run `deepsphinx-infer`. Options need to be the same as
that used for training session for the checkpoint

Pretrained model
---------------

[https://s3.amazonaws.com/deepsphinx/batch-21937.data-00000-of-00001](https://s3.amazonaws.com/deepsphinx/batch-21937.data-00000-of-00001)
[https://s3.amazonaws.com/deepsphinx/batch-21937.index](https://s3.amazonaws.com/deepsphinx/batch-21937.index)

