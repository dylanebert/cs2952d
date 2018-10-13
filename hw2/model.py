#!/usr/bin/env python3

import math
import random
from collections import defaultdict


class Model:
    """
    This is the class for our model which includes the constructed grammar. Note that this will
    be used to train our parsing by ranking each parse.
    """

    def __init__(self, grammar, executor=None):
        """
        :param grammar: Grammar defined for this model
        :param executor: A function that can evaluate our type of parse
        """
        self.grammar = grammar
        self.executor = executor

    def parse_input(self, input):
        """
        This method will evalute the parses for a specific input and rank them according to the score
        :param input: Input string
        :return: A ranked list of parses
        """
        # This is the CYK that was implemented
        parses = self.grammar.parse_input(input)
        for parse in parses:
            # Evaluate the parse if given an executor
            parse.denotation = self.executor(parse.semantics)
        return parses
