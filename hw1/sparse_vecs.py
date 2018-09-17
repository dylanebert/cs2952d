#!/usr/bin/env python3
""" sparse_vecs.py

    Run this script (use Python 3!) with the --help flag to see how to use command-line options.

    Hint: Anything already imported for you might be useful, but not necessarily required, to use :)
"""

import argparse
import collections

import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine

import utils

""" hyperparameters ( do not modify! ) """
WINDOW_SIZE = 2
NUM_CLOSEST = 20

def word_vectors(corpus, vocab):
    """
        Input: A corpus (list of list of string) and a vocab (word-to-id mapping)
        Output: A lookup table that maps [word id] -> [word vector]
    """
    id2idx = {}
    vectors = {}
    for i, key in enumerate(list(vocab.values())):
        if i % 1000 == 0:
            print('{0} of {1}'.format(i, len(vocab)), end='\r')
        id2idx[key] = i
        vectors[key] = np.zeros((len(vocab),))
        i += 1
    print('Finished building empty word vectors')
    for j, s in enumerate(corpus):
        if j % 1000 == 0:
            print('{0} of {1}'.format(j, len(corpus)), end='\r')
        for i in range(len(s)):
            idx = vocab[s[i]]
            for j in range(1, WINDOW_SIZE):
                if i - j >= 0:
                    vectors[idx][id2idx[vocab[s[i-j]]]] += 1
                if i + j < len(s):
                    vectors[idx][id2idx[vocab[s[i+j]]]] += 1
    print('Finished computing word vectors')
    return vectors

def closest(lookup_table, wordvec):
    closest = [w[0] for w in sorted(lookup_table.items(), key=lambda x: cosine(wordvec, x[1]))[:NUM_CLOSEST]]
    return closest

def main():
    """
    Task: Transform a corpus of text into word vectors according to this context-window principle.
    1. Load the data - this is done for you.
    2. Construct a vocabulary across the entire corpus. This should map a string (word) to id.
    3. Use the vocabulary (as a word-to-id mapping) and corpus to construct the sparse word vectors.
    """
    sentences = utils.load_corpus(args.corpus)
    word_freqs = utils.word_counts(sentences)
    sentences, word_freqs = utils.trunc_vocab(sentences, word_freqs)
    vocab, inverse_vocab = utils.construct_vocab(sentences)
    lookup_table = word_vectors(sentences, vocab)

    #1 - nearest to my favorite word
    fav_word = 'Texas'
    nearest_ids = closest(lookup_table, lookup_table[vocab[fav_word]])
    nearest_words = [inverse_vocab[i] for i in nearest_ids]
    print('Nearest to {0}: {1}'.format(fav_word, nearest_words))

    #2/3 - most and least similar
    words = list(vocab.keys())
    most_similar = (None, None, .5)
    least_similar = (None, None, .5)
    print('Computing most and least similar pairs')
    for i in range(len(words) - 1):
        print('{0} of {1}'.format(i, len(words)), end='\r')
        for j in range(i+1, len(words)):
            w1 = words[i]
            w2 = words[j]
            sim = 1 - cosine(lookup_table[vocab[w1]], lookup_table[vocab[w2]])
            if sim > most_similar[2]:
                most_similar = (w1, w2, sim)
            if sim < least_similar[2]:
                least_similar = (w1, w2, sim)
    print('Finished computing most and least similar pairs')
    print('Most similar pair: {0}, {1}'.format(most_similar[0], most_similar[1]))
    print('Least similar pair: {0}, {1}'.format(least_similar[0], least_similar[1]))

if __name__ == "__main__":
    # NOTE: feel free to add your own arguments here, but we will only ever run your script
    #  based on the arguments provided in this stencil
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help="path to corpus file", default="corpus.txt")
    args = parser.parse_args()

    np.set_printoptions(linewidth=150)

    main()
