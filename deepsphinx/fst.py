'''Functions for FST calculation'''
from itertools import groupby
import numpy as np

from deepsphinx.vocab import VOCAB_TO_INT, VOCAB_SIZE

import pickle


def combine(probs):
    ''' Add various log probabilities'''
    if len(probs) == 0:
        return np.float32(-50.0)
    probs_max = np.max(probs)
    return probs_max + np.log(np.sum(np.exp(probs - probs_max)))

backoff = VOCAB_TO_INT['<backoff>']

def next_fst(fst):
    try:
        return pickle.load(open('next.p','rb'), encoding='latin1')
    except:
        pass
    ret = {}
    for s in fst.states():
        for a in fst.arcs(s):
            ret[(s, a.ilabel)] = (a.nextstate, -np.float32(a.weight.to_string()))
    pickle.dump(ret, open('next.p', 'wb'))
    return ret

def allnext(nfst, s, voc, probtill=0.0):
    ret = []
    ss = s
    while 1:
        if (ss, voc) in nfst:
            t=nfst[(ss, voc)]
            ret.append((t[0], probtill + t[1]))
        if (ss, backoff) in nfst:
            probtill += nfst[(ss, backoff)][1]
            ss = nfst[(ss, backoff)][0]
        else:
            break
    return ret


def probs_fst(nfst, N):
    try:
        return pickle.load(open('probs.p','rb'), encoding='latin1')
    except:
        pass
    ret = np.zeros((N, VOCAB_SIZE), dtype='float32')
    for s in range(N):
        for voc in range(VOCAB_SIZE):
            ret[s][voc] = combine([a[1] for a in allnext(nfst, s, voc)])
    pickle.dump(ret, open('probs.p', 'wb'))
    return ret

def in_fst(nfst, pfst, text):
    '''Checks weather a text is possible in fst'''
    states = [0]
    probs = [0.0]
    num_st = [1]
    ret = True
    scores = 0.0
    states, probs, num_st, vocab_probs = fst_cost_single(
        states, probs, num_st, VOCAB_TO_INT['<s>'], nfst, pfst, 5)
    vocab_probs += combine(vocab_probs)
    for i in text:
        scores += vocab_probs[i]
        states, probs, num_st, vocab_probs = fst_cost_single(
            states, probs, num_st, i, nfst, pfst, 5)
        vocab_probs += combine(vocab_probs)
        if num_st == [0]:
            ret = False
    #print('LM loss: ', scores / len(text), ret)
    return ret

def fst_cost_single(poss_states, probs, num_fst_states, inp, nfst, pfst, max_states):
    '''Calculate next state and their probabilites given an input symbol and
    current possible states and their probabilites'''
    # A very crude try
    if num_fst_states[0] < 1:
        return poss_states, probs, num_fst_states, np.zeros(VOCAB_SIZE, dtype='float32')
    num_fst_states = num_fst_states[0]
    next_tup = {}
    for i in range(num_fst_states):
        for (s, w) in  allnext(nfst, poss_states[i], inp, probs[i]):
            if s in next_tup:
                next_tup[s].append(w)
            else:
                next_tup[s] = [w]
    num_states = 0
    next_states = np.zeros((max_states), dtype='int32')
    next_probs = np.zeros((max_states), dtype='float32')
    probs_per = np.zeros((max_states, VOCAB_SIZE), dtype='float32')
    probs_arr = np.zeros((VOCAB_SIZE), dtype='float32')
    for state, weight in next_tup.items():
        next_states[num_states] = state
        prob = combine(weight)
        next_probs[num_states] = prob
        probs_per[num_states] = prob + pfst[state]
        num_states += 1
    next_probs -= combine(next_probs[:num_states])
    for i in range(VOCAB_SIZE):
        probs_arr[i] = combine(probs_per[:num_states, i])
    probs_arr -= combine(probs_arr)
    return next_states, next_probs, np.asarray([num_states], dtype='int32'), probs_arr

#def fst_costs(states, probs, num, inputs, fst, max_states):
#    '''Calculate next state and their probabilites given an input symbol and
#    current possible states and their probabilites for a batch'''
#    next_states = np.zeros_like(states)
#    next_probs = np.zeros_like(probs)
#    next_num = np.zeros_like(num)
#    scores = np.zeros((num.shape[0], VOCAB_SIZE), 'float32')
#    inp = []
#    for i in range(num.shape[0]):
#        inp.append((states[i], probs[i], num[i][0], inputs[i], fst, max_states))
#    results = list(map(lambda x: fst_cost_single(*x), inp))
#    for i in range(num.shape[0]):
#        next_states[i], next_probs[i], next_num[i][0], scores[i] = results[i]
#    return next_states, next_probs, next_num, scores
