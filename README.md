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

![model](images/model.jpg)


Data format
-----------

Data contains transcripts file and audio in flac format. The format of
transcription file is `<transcript>\<set_id>\<speaker_id>\<file path>` for each
transcript. Transcripts are separated by newline.

Usage
-----

Install `tensorflow`, `python_speech_features` and `openfst`(optional) first before
running, and generate FST if set in command line options.


```
usage: deepsphinx_train [-h] [--nfilt NFILT] [--keep_prob KEEP_PROB]
                        [--max_output_len MAX_OUTPUT_LEN]
                        [--rnn_size RNN_SIZE] [--num_layers NUM_LAYERS]
                        [--num_decoding_layers NUM_DECODING_LAYERS]
                        [--batch_size BATCH_SIZE]
                        [--learning_rate LEARNING_RATE]
                        [--num_epochs NUM_EPOCHS] [--beam_width BEAM_WIDTH]
                        [--learning_rate_decay LEARNING_RATE_DECAY]
                        [--min_learning_rate MIN_LEARNING_RATE]
                        [--display_step DISPLAY_STEP] [--data_dir DATA_DIR]
                        [--job_dir JOB_DIR]
                        [--checkpoint_path CHECKPOINT_PATH]
                        [--best_n_inference BEST_N_INFERENCE]
                        [--eval_only EVAL_ONLY] [--fst_path FST_PATH]
                        [--cutoff_range CUTOFF_RANGE]
                        [--trans_file TRANS_FILE]
                        [--use_inference_lm [USE_INFERENCE_LM]]
                        [--nouse_inference_lm] [--use_train_lm [USE_TRAIN_LM]]
                        [--nouse_train_lm]

optional arguments:
  -h, --help            show this help message and exit
  --nfilt NFILT         Number of filters in mel specta
  --keep_prob KEEP_PROB
                        Keep probability for dropout
  --max_output_len MAX_OUTPUT_LEN
                        Maximum output length for Beam Search
  --rnn_size RNN_SIZE   Size of LSTM
  --num_layers NUM_LAYERS
                        Number of LSTM Layers for encoding layer
  --num_decoding_layers NUM_DECODING_LAYERS
                        Number of LSTM Layers for decoding layer
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
  --job_dir JOB_DIR     Directory in which summary and checkpoint is stored
  --checkpoint_path CHECKPOINT_PATH
                        Load a trained model
  --best_n_inference BEST_N_INFERENCE
                        Take best of n for beam search
  --eval_only EVAL_ONLY
                        Only evaluate. --checkpoint-path is required
  --fst_path FST_PATH   Path of language FST
  --cutoff_range CUTOFF_RANGE
                        Attention cutoff from previous mean range
  --trans_file TRANS_FILE
                        Path of transcription file
  --use_inference_lm [USE_INFERENCE_LM]
                        Use LM during inference
  --nouse_inference_lm
  --use_train_lm [USE_TRAIN_LM]
                        Use LM during training
  --nouse_train_lm
```

Language model
--------------

See file `make_fst` for commands for generating Language model FST.

Current results
---------------

I am getting round 15% WER in dev93 by training in si284 set.
