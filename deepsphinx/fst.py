'''Functions for FST calculation'''
from itertools import groupby
import numpy as np

from deepsphinx.vocab import VOCAB_TO_INT, VOCAB_SIZE

def in_fst(fst, text):
    '''Checks weather a text is possible in fst'''
    states = [0]
    probs = [0.0]
    num_st = 1
    ret = True
    scores = 0.0
    states, probs, num_st, vocab_probs = fst_cost_single(
        states, probs, num_st, VOCAB_TO_INT['<s>'], fst, 5)
    vocab_probs -= combine(vocab_probs)
    for i in text:
        scores += vocab_probs[i]
        states, probs, num_st, vocab_probs = fst_cost_single(
            states, probs, num_st, i, fst, 5)
        vocab_probs -= combine(vocab_probs)
        if num_st == 0:
            ret = False
    # print('LM loss: ', scores / len(text), ret)
    return ret


def dfs_probs(state, prob, fst, probs_arr):
    '''Find all possible next states'''
    for arc in fst.arcs(state):
        cur_prob = prob - np.log(10) * np.float32(arc.weight.to_string())
        if arc.ilabel == VOCAB_TO_INT['<backoff>']:
            dfs_probs(arc.nextstate, cur_prob, fst, probs_arr)
            continue
        probs_arr[arc.ilabel].append(cur_prob)

def dfs_find(state, prob, fst, inp, out):
    '''Find next possible states with an input symbol'''
    for arc in fst.arcs(state):
        cur_prob = prob - np.log(10) * np.float32(arc.weight.to_string())
        if arc.ilabel == VOCAB_TO_INT['<backoff>']:
            dfs_find(arc.nextstate, cur_prob, fst, inp, out)
            continue
        if arc.ilabel == inp:
            out.append((arc.nextstate, cur_prob))


def combine(probs):
    ''' Add various log probabilities'''
    if len(probs) == 0:
        return np.float32(-50.0)
    probs_max = np.max(probs)
    return probs_max + np.log(np.sum(np.exp(probs - probs_max)))


def fst_cost_single(poss_states, probs, num_fst_states, inp, fst, max_states):
    '''Calculate next state and their probabilites given an input symbol and
    current possible states and their probabilites'''
    # A very crude try
    if num_fst_states >= 1:
        probs[:num_fst_states] -= combine(probs[:num_fst_states])
    next_tup = []
    for i in range(num_fst_states):
        dfs_find(poss_states[i], probs[i], fst, inp, next_tup)
    next_tup = groupby(sorted(next_tup), lambda t: t[0])
    num_states = 0
    next_states = []
    next_probs = []
    probs_arr = [[] for _ in range(VOCAB_SIZE)]
    for state, weight in next_tup:
        num_states += 1
        next_states.append(state)
        prob = combine(np.asarray([w[1] for w in weight]))
        next_probs.append(prob)
        dfs_probs(state, prob, fst, probs_arr)
    for i in range(VOCAB_SIZE):
        probs_arr[i] = combine(probs_arr[i])
    while len(next_states) < max_states:
        next_states.append(0)
        next_probs.append(0)
    return np.asarray(next_states), np.asarray(next_probs), num_states, np.asarray(probs_arr)

def fst_costs(states, probs, num, inputs, fst, max_states):
    '''Calculate next state and their probabilites given an input symbol and
    current possible states and their probabilites for a batch'''
    next_states = np.zeros_like(states)
    next_probs = np.zeros_like(probs)
    next_num = np.zeros_like(num)
    scores = np.zeros((num.shape[0], VOCAB_SIZE), 'float32')
    for i in range(num.shape[0]):
        next_states[i], next_probs[i], next_num[i][0], scores[i] = fst_cost_single(states[i],
                                                                                   probs[i],
                                                                                   num[i][0],
                                                                                   inputs[i],
                                                                                   fst,
                                                                                   max_states)
    return next_states, next_probs, next_num, scores
