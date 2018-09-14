#!/usr/bin/env python3
""" sparse_vecs.py

    Run this script (use Python 3!) with the --help flag to see how to use command-line options.

    Hint: Anything already imported for you might be useful, but not necessarily required, to use :)
"""

import argparse
import collections

import numpy as np
from scipy.spatial.distance import pdist, squareform

import utils

""" hyperparameters ( do not modify! ) """
WINDOW_SIZE = 2
NUM_CLOSEST = 20

def word_vectors(corpus, vocab):
    """
        Input: A corpus (list of list of string) and a vocab (word-to-id mapping)
        Output: A lookup table that maps [word id] -> [word vector]
    """
    raise NotImplementedError("word_vectors")

def most_similar(lookup_table, wordvec):
    """ Helper function (optional).

        Given a lookup table and word vector, find the top most-similar word ids to the given
        word vector. You can limit this to the first NUM_CLOSEST results.
    """
    raise NotImplementedError("most_similar")

def main():
    """
    Task: Transform a corpus of text into word vectors according to this context-window principle.
    1. Load the data - this is done for you.
    2. Construct a vocabulary across the entire corpus. This should map a string (word) to id.
    3. Use the vocabulary (as a word-to-id mapping) and corpus to construct the sparse word vectors.
    """
    sentences = utils.load_corpus(args.corpus)
    vocab, inverse_vocab = utils.construct_vocab(sentences)
    lookup_table = word_vectors(sentences, vocab)

    """ TODO: Implement what you need to answer the writeup questions. """


if __name__ == "__main__":
    # NOTE: feel free to add your own arguments here, but we will only ever run your script
    #  based on the arguments provided in this stencil
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, help="path to corpus file", default="corpus.txt")
    args = parser.parse_args()

    np.set_printoptions(linewidth=150)

    main()

