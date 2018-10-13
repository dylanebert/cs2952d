#!/usr/bin/env python3

class Example:
    """
    Class to store data for parsing.
    """
    def __init__(self, input=None, parse=None, semantics=None, denotation=None):
        self.input = input
        self.parse = parse
        self.semantics = semantics
        self.denotation = denotation

    def __str__(self):
        fields = []
        self.input != None and fields.append('input=\'%s\'' % self.input.replace("'", "\'"))
        self.parse != None and fields.append('parse=%s' % self.parse)
        self.semantics != None and fields.append('semantics=%s' % str(self.semantics))
        self.denotation != None and fields.append('denotation=%s' % str(self.denotation))
        return "Example({})".format(", ".join(fields))