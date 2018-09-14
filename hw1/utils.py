import os

import collections
from tqdm import tqdm
import numpy as np

import hyperparams as hp

def construct_vocab(corpus):
    """
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple of dicts: (vocab, inverse_vocab)
                vocab: A dict mapping str -> int. This will be your vocabulary.
                inverse_vocab: Inverse mapping int -> str
    """
    words = []
    vocab = {}
    inverse_vocab = {}
    i = 0
    for s in corpus:
        for w in s:
            if w not in words:
                vocab[w] = i
                inverse_vocab[i] = w
                i += 1
    return (vocab, inverse_vocab)

def trunc_vocab(corpus, counts):
    """ Limit the vocabulary to the 10k most-frequent words. Remove rare words from
         the original corpus.
        Input: A list of list of string. Each string represents a word token.
        Output: A tuple (new_corpus, new_counts)
                new_corpus: A corpus (list of list of string) with only the 10k most-frequent words
                new_counts: Counts of the 10k most-frequent words

        Hint: Sort the keys of counts by their values
    """
    frequent = [w[0] for w in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10000]]
    new_corpus = [[w for w in s if w in frequent] for s in corpus]
    new_counts = {k: v for k, v in counts.items() if k in frequent}
    return (new_corpus, new_counts)

def load_corpus(path):
    """ Reads the data from disk.
        Returns a list of sentences, where each sentence is split into a list of word tokens
    """
    with open(path, 'r', encoding='utf-8') as f:
        c = [line.split() for line in f]
    return c

def word_counts(corpus):
    """ Given a corpus (such as returned by load_corpus), return a dictionary
        of word frequencies. Maps string token to integer count.
    """
    return collections.Counter(w for s in corpus for w in s)

def keep_prob(word_prob):
    """
        Probability of keeping a word, as a function of that word's probability
        in the corpus: word_prob = word_freq/total_words_in_corpus
    """
    return (np.sqrt(word_prob/hp.T_SAMPLE) + 1)*(hp.T_SAMPLE/word_prob)

def subsample(corpus, freqs):
    """
        Implements subsampling of frequent words (Mikolov et al., 2013), as they tend to carry less
        information value than rare words.
          Input: a corpus (as returned by load_corpus) and a dictionary of normalised word frequencies
          Output: the corpus with words dropped (in-place) according to keep_prob
    """
    N = sum(f for f in freqs.values())
    for ix,sentence in tqdm(enumerate(corpus), desc="Subsampling"):
        corpus[ix] = list(filter(lambda w: np.random.rand() < keep_prob(freqs[w]/N), sentence))
    return corpus

def load_data(restore_path):
    dirname = os.path.dirname(restore_path)
    with open(os.path.join(dirname, "data.pkl"), "rb") as f:
        vocab, inverse_vocab = pickle.load(f)
        return vocab, inverse_vocab

def save_data(save_path, vocab, inverse_vocab):
    with open(os.path.join(save_path, "data.pkl"), "wb") as f:
        pickle.dump((vocab, inverse_vocab), f)
