import tensorflow as tf
import pywrapfst as fst
from itertools import groupby
import numpy as np

from vocab import vocab_to_int, vocab_size
from utils import FileOpen

def in_fst(f, text):
    st = [0]
    probs = [0.0]
    num_st = 1
    ret = True
    sc = 0.0
    st, probs, num_st, vocab_probs = fstCostSingle(st, probs, num_st, vocab_to_int['<s>'], f, 5)
    vocab_probs -= combine(vocab_probs)
    for i in text:
        sc += vocab_probs[i]
        st, probs, num_st, vocab_probs = fstCostSingle(st, probs, num_st, i, f, 5)
        vocab_probs -= combine(vocab_probs)
        if (num_st == 0):
            ret = False
    # print("LM loss: ", sc / len(text), ret)
    return ret


def dfsProbs(state, prob, LMfst, probsArr):
    for arc in LMfst.arcs(state):
        curProb = prob - np.log(10) * np.float32(arc.weight.to_string())
        if arc.ilabel == vocab_to_int['<backoff>']:
            dfsProbs(arc.nextstate, curProb, LMfst, probsArr)
            continue
        probsArr[arc.ilabel].append(curProb)

def dfsFind(state, prob, LMfst, inp, out):
    for arc in LMfst.arcs(state):
        curProb = prob - np.log(10) * np.float32(arc.weight.to_string())
        if arc.ilabel == vocab_to_int['<backoff>']:
            dfsFind(arc.nextstate, curProb, LMfst, inp, out)
            continue
        if arc.ilabel == inp:
            out.append((arc.nextstate, curProb))

def combine(probs):
    if (len(probs) == 0):
        return np.float32(-50.0)
    probs_max = np.max(probs)
    return probs_max + np.log(np.sum(np.exp(probs - probs_max)))

def fstCostSingle(poss_states, probs, num_fst_states, inp, LMfst, max_states):
    # A very crude try
    probs_sum = combine(probs[:num_fst_states])
    if num_fst_states >= 1:
        probs[:num_fst_states] -= combine(probs[:num_fst_states])
    next_tup = []
    for i in range(num_fst_states):
        dfsFind(poss_states[i], probs[i], LMfst, inp, next_tup)
    next_tup = groupby(sorted(next_tup), lambda t: t[0])
    num_states = 0
    next_states = []
    next_probs = []
    scores = []
    probsArr = [[] for _ in range(vocab_size)]
    for st, we in next_tup:
        num_states += 1
        next_states.append(st)
        prob = combine(np.asarray([w[1] for w in we]))
        next_probs.append(prob)
        dfsProbs(st, prob, LMfst, probsArr)
    for i in range(vocab_size):
        probsArr[i] = combine(probsArr[i])
    while(len(next_states) < max_states):
        next_states.append(0)
        next_probs.append(0)
    return np.asarray(next_states), np.asarray(next_probs), num_states, np.asarray(probsArr)

def fstCosts(states, state_probs, num_fst_states, inputs, LMfst, max_states):
    next_states = np.zeros_like(states)
    next_state_probs = np.zeros_like(state_probs)
    next_num_states = np.zeros_like(num_fst_states)
    scores = np.zeros((num_fst_states.shape[0], vocab_size), 'float32')
    for i in range(num_fst_states.shape[0]):
        next_states[i], next_state_probs[i], next_num_states[i][0], scores[i] = fstCostSingle(states[i],
                state_probs[i], num_fst_states[i][0], inputs[i], LMfst, max_states)
    return next_states, next_state_probs, next_num_states, scores
