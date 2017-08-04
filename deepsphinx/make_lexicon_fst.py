# While I could get this functionality by some other codes they are much more
# complicated than what I needed and required special file formats. Also I can
# readily share symbol ids with acoustic and language model
import sys
from data import vocab_to_int
import pywrapfst as fst

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: {} <sym>".format(__file__))
        sys.exit(1)
    start_symbol = 0
    curr_symbol = 1
    for word in open(sys.argv[1]).readlines():
        word, word_sym = word.strip().split()
        if word == '<eps>':
            continue
        if word == '<backoff>' or word == '<s>' or word == '</s>':
            print("{}\t{}\t{}\t{}".format(start_symbol, start_symbol,
                vocab_to_int[word], word_sym))
            continue
        word = word.upper()
        print("{}\t{}\t{}\t{}".format(start_symbol, curr_symbol,
            vocab_to_int[word[0]], word_sym))
        for c in word[1:]:
            print("{}\t{}\t{}\t{}".format(curr_symbol, curr_symbol + 1,
                vocab_to_int[c], vocab_to_int['<eps>']))
            curr_symbol += 1
        # This disambiguation is enough to determinize it
        print("{}\t{}\t{}\t{}".format(curr_symbol, start_symbol, vocab_to_int[' '],
            vocab_to_int['<eps>']))
        curr_symbol += 1
    print(start_symbol)
