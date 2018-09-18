#!/usr/bin/env python3
""" word2vec.py

    Run this script (use Python 3!) with the --help flag to see how to use command-line options.

    Hint: Anything already imported for you might be useful, but not necessarily required, to use :)
"""
import os
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

from tqdm import tqdm

import utils
import hyperparams as hp

class SkipGramNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(SkipGramNetwork, self).__init__()
        self.embeddings = nn.Linear(vocab_size, embedding_size)
        self.logits = nn.Linear(embedding_size, vocab_size)
        self.onehot_lookup = torch.eye(vocab_size, vocab_size)

    def forward(self, input):
        input = self.onehot_lookup[input].to(device)
        h = F.relu(self.embeddings(input))
        logits = self.logits(h)
        return F.log_softmax(logits, dim=1)

def skip_grams(corpus, vocab):
    data = []
    for s in corpus:
        for i in range(len(s)):
            for j in range(1, hp.WINDOW_SIZE):
                if i - j >= 0:
                    data.append((vocab[s[i]], vocab[s[i-j]]))
                if i + j < len(s):
                    data.append((vocab[s[i]], vocab[s[i+j]]))
    return data

def train(model, dataloader):
    """ Complete this function. Return a history of loss function evaluations """

    loss_function = nn.NLLLoss() # optionally, you can use nn.CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARN_RATE)
    loss_history = [] # NOTE: use this
    for epoch in range(hp.NUM_EPOCHS):
        print("---- Epoch {} of {} ----".format(epoch+1, hp.NUM_EPOCHS))
        loss_per_epoch = 0
        num_batches = 0
        for batch_idx, (target, context) in enumerate(dataloader):
            target = torch.LongTensor(target).to(device)
            context = torch.LongTensor(context).to(device)

            model.zero_grad() # clear gradients (torch will accumulate them)
            probs = model(target)
            loss = loss_function(probs, context)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if (batch_idx % 500) == 0:
                print("\t Batch {}".format(batch_idx))

        ckpt = os.path.join(args.save, "model-epoch{}.ckpt".format(epoch))
        torch.save(model.state_dict(), ckpt)
        print("Checkpoint saved to {}".format(ckpt))

    return loss_history

def closest(lookup_table, wordvec):
    closest = [w[0] for w in sorted(enumerate(lookup_table), key=lambda x: cosine(wordvec, x[1]))[:hp.NUM_CLOSEST]]
    return closest

def main():
    """ Task: Train a neural network to predict skip-grams.
        1. Load the data, this is done for you.
        2. Limit the vocabulary to 10,000 words. Use the provided mapping of word-counts to keep only the most-frequent words.
        3. Generate skipgrams from the input. As before, you will need to build a vocabulary to map words to integer ids.
        4. Implement the training loop for your neural network.
        5. After training, index into the rows of the embedding matrix using a word ID.
        6. Use t-SNE (or other dimensionality reduction algorithm) to visualise the embedding manifold on a subset of data.
    """

    net = SkipGramNetwork(hp.VOCAB_SIZE, hp.EMBED_SIZE).to(device)
    print(net)

    if args.restore:
        net.load_state_dict(torch.load(args.restore))
        vocab, inverse_vocab = utils.load_data(args.restore)
        print("Model restored from disk.")
    else:
        sentences = utils.load_corpus(args.corpus)
        word_freqs = utils.word_counts(sentences)
        sentences, word_freqs = utils.trunc_vocab(sentences, word_freqs)
        sentences = utils.subsample(sentences, word_freqs)

        vocab, inverse_vocab = utils.construct_vocab(sentences)
        skipgrams = skip_grams(sentences, vocab)
        utils.save_data(args.save, vocab, inverse_vocab)

        loader = DataLoader(skipgrams, batch_size=hp.BATCH_SIZE, shuffle=True)
        loss_hist = train(net, loader)
        print(loss_hist)

        plt.plot(loss_hist)
        plt.show()

    # the weights of the embedding matrix are the lookup table
    lookup_table = net.embeddings.weight.data.cpu().numpy()
    lookup_table = np.transpose(lookup_table) #transpose the table so we can index directly

    #2 - nearest to favorite
    fav_word = 'Texas'
    wordvec = lookup_table[vocab[fav_word]]
    nearest = closest(lookup_table, wordvec)
    nearest_words = [inverse_vocab[w] for w in nearest if w in inverse_vocab]
    print('Nearest to {0}: {1}'.format(fav_word, nearest_words))

    #3 - most and least similar
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

    #4 - manifold
    print('Fitting to manifold')
    manifold = TSNE().fit_transform(lookup_table)
    plt.scatter(manifold[:,0], manifold[:,1])
    plt.show()

if __name__ == "__main__":
    # NOTE: feel free to add your own arguments here, but we will only ever run your script
    #  based on the arguments provided in this stencil
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str, help="path to corpus file")
    parser.add_argument("--device", help="pass --device cuda to run on gpu", default="cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", help="path to save directory", default="saved_runs")
    group.add_argument("--restore", help="path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    np.set_printoptions(linewidth=150)
    device = torch.device('cpu')
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    print("Using device {}".format(device))

    main()
