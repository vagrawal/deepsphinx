Some parts are curently wrong in this file

Introduction
------------

This repo contains an end-to-end speech recognition system implented using
tensorflow. It uses bidirectional LSTMs in the encoding layer with attention
in the decoding layer. The LSTMs in the encoding layer are strided and in every
layer, the time dimension is reduced by 2. The complete model looks much like
one from [1].

Model
-----

Default configuration contains 3 bidirectional layers for encoding and decoding
layer:

![model](images/model.jpeg)


Data format
-----------

Data contains transcript file and audio in flac format. The format of
transcription file is `<transcript>\<set_id>\<speaker_id>\<file path>`

Usage
-----

Install `tensorflow`, `python_speech_features` and `openfst` first before
running, and generate FST if set in command line options.


```
usage: deepsphinx_train [-h] [--nfilt NFILT] [--keep_prob KEEP_PROB]
                        [--max_output_len MAX_OUTPUT_LEN]
                        [--rnn_size RNN_SIZE] [--num_layers NUM_LAYERS]
                        [--batch_size BATCH_SIZE]
                        [--learning_rate LEARNING_RATE]
                        [--num_epochs NUM_EPOCHS] [--beam_width BEAM_WIDTH]
                        [--learning_rate_decay LEARNING_RATE_DECAY]
                        [--min_learning_rate MIN_LEARNING_RATE]
                        [--display_step DISPLAY_STEP] [--data_dir DATA_DIR]
                        [--job-dir JOB_DIR]
                        [--checkpoint-path CHECKPOINT_PATH]
                        [--best-n-inference BEST_N_INFERENCE]
                        [--eval-only EVAL_ONLY] [--fst-path FST_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --nfilt NFILT         Number of filters in mel specta
  --keep_prob KEEP_PROB
                        Keep probability for dropout
  --max_output_len MAX_OUTPUT_LEN
                        Maximum output length for Beam Search
  --rnn_size RNN_SIZE   Size of LSTM
  --num_layers NUM_LAYERS
                        Number of LSTM Layers
  --batch_size BATCH_SIZE
                        Batch size of data
  --learning_rate LEARNING_RATE
                        Learning rate
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --beam_width BEAM_WIDTH
                        Beam width. Must be lesser than vocab size
  --learning_rate_decay LEARNING_RATE_DECAY
                        Learning rate decay
  --min_learning_rate MIN_LEARNING_RATE
                        Minimum learning rate
  --display_step DISPLAY_STEP
                        Check training loss after every display_step batches
  --data_dir DATA_DIR   Directory of data
  --job-dir JOB_DIR     Directory in which summary and checkpoint is stored
  --checkpoint-path CHECKPOINT_PATH
                        Load a trained model
  --best-n-inference BEST_N_INFERENCE
                        Take best of n for beam search
  --eval-only EVAL_ONLY
                        Only evaluate. --checkpoint-path is required
  --fst-path FST_PATH   Path of language FST```

Current results
---------------

I am getting round 15% WER in dev93 by training in si_284 set.


[1]: Dzmitry Bahdanau, Jan Chorowski, Dmitriy Serdyuk,
Philemon Brakel, and Yoshua Bengio, “End-to-end
attention-based large vocabulary speech recognition,”
arXiv preprint arXiv:1508.04395, 2015
