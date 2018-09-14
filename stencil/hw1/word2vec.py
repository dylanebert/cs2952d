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
import matplotlib.pyplot as plt

from tqdm import tqdm

import utils
import hyperparams as hp

class SkipGramNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        """ TODO
            Initialise the layers of your network here. You should have two layers:
                - an embedding lookup layer
                - a linear (fully-connected) layer
        """
        super(SkipGramNetwork, self).__init__()
        raise NotImplementedError("SkipGramNetwork.__init__")

    def forward(self, inputs):
        """ TODO
            Implement the forward pass for your network.
             inputs: batch_size x 1 tensor, each batch element will be a word id
             outputs: batch_size x vocab_size tensor

            You shouldn't need to use any non-linearities.
            If using NLLLoss don't forget to apply a log softmax to your output.
        """
        raise NotImplementedError("SkipGramNetwork.forward")

def skip_grams(corpus, vocab):
    """ TODO
        Input: A corpus (list of list of string) and a vocab (word-to-id mapping)
        Output: A list of skipgrams (each skipgram should be a list of length 2)
    """
    raise NotImplementedError("skip_grams")

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
            """ TODO:
                1. perform the forward pass
                2. evaluate the loss given output from forward pass and known context word
                3. backward pass
                4. parameter update
            """
            raise NotImplementedError("training loop")

            if (batch_idx % 500) == 0:
                print("\t Batch {}".format(batch_idx))

        ckpt = os.path.join(args.save, "model-epoch{}.ckpt".format(epoch))
        torch.save(model.state_dict(), ckpt)
        print("Checkpoint saved to {}".format(ckpt))

    return loss_history

def most_similar(lookup_table, wordvec):
    """ TODO
        Given a lookup table and word vector, find the top-most
        similar word ids to the given word vector. You may limit this to the first
        NUM_CLOSEST results.
    """
    raise NotImplementedError("most_similar")

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
        sentences, word_freqs = utils.trunc_vocab(sentences, word_freqs) # TODO
        sentences = utils.subsample(sentences, word_freqs)

        vocab, inverse_vocab = utils.construct_vocab(sentences) # TODO
        skipgrams = skip_grams(sentences, vocab) # TODO
        utils.save_data(args.save, vocab, inverse_vocab)

        loader = DataLoader(skipgrams, batch_size=hp.BATCH_SIZE, shuffle=True)
        loss_hist = train(net, loader) # TODO returns loss function evaluations as python list

        """ You can plot loss_hist for your writeup:
            plt.plot(loss_hist)
            plt.show()
        """

    # the weights of the embedding matrix are the lookup table
    lookup_table = net.embeddings.weight.data.cpu().numpy()

    """ TODO: Implement what you need in order to answer the writeup questions. """


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
