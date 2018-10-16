#!/usr/bin/env python3

# XXX: Scroll down to the bottom for the definition of main

import math
from types import FunctionType
from collections import defaultdict
from itertools import product
from expressions import *
import nltk

MAX_CELL_CAPACITY = 1000  # upper bound on number of parses in one chart cell

#=========================================== Rule ====================================================

class Rule:
    """
    This is the Rule class.
    """
    def __init__(self, lhs, rhs, sem=None):
        """
        :param lhs: The rule type (i.e. $E, $UnOp, $BinOp, $EBO)
        :param rhs: The rule ....
        :param sem: The semantic representation (i.e. "+", Expression)
        """
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem

    def is_cat(self, label):
        """
        Return true if the label represents a category.
        :param label: string like $E
        """
        return label[0] == '$'

    def is_lexical(self):
        """
        Returns true if this is a lexical rule. Hint: use is_cat
        """
        for item in self.rhs:
            if self.is_cat(item):
                return False
        return True

    def is_binary(self):
        """
        Returns true if the rule is binary
        """
        return len(self.rhs) == 2 and self.is_cat(self.rhs[0]) and self.is_cat(self.rhs[1])

    def apply_semantics(self, sems):
        # Note that this function would not be needed if we required that semantics
        # always be functions, never bare values.  That is, if instead of
        # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
        # But that would be cumbersome.
        if isinstance(self.sem, FunctionType):
            return self.sem(sems)
        else:
            return self.sem

    ############## Part 2 ##############

    def is_unary(self):
        """
        Returns true if the rule is unary
        """
        return len(self.rhs) == 1 and self.is_cat(self.rhs[0])

    def is_optional(self, label):
        """
        Returns true iff the given RHS item is optional, i.e., is marked with an
        initial '?'.
        :param label: string like ?$Optionals
        """
        return label[0] == '?'

    def contains_optionals(self):
        """
        Returns true iff the given Rule contains any optional items on the RHS.
        """
        for item in self.rhs:
            if self.is_optional(item):
                return True
        return False

    def __str__(self):
        """
        String representation of a rule
        """
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

#=========================================== Parse ====================================================

class Parse:
    """
    This is a simple container class that uses the given rule to compute the semantics of its
    given children. If the given rule is lexical, the children are simply the tokens. If the
    rule is binary then the children are other Parses.
    """
    def __init__(self, rule, children):
        """
        :param rule: The Rule to compute the semantics
        :param children: List of tokens/Parses to apply our rule to and compute the semantics
        """
        self.rule = rule
        self.children = tuple(children[:])
        self.denotation = None # Do not worry about this in part 1

        # This will store our semantic parse
        self.semantics = self.compute_semantics()

    def __str__(self):
        """
        Just a nice way of printing our final parses.
        """
        return '({}, {})'.format(self.rule.lhs, ' '.join([str(c) for c in self.children]))

    def compute_semantics(self):
        """
        This is where we will apply the semantics that we defined for our rules earlier.
        Hint: Remember when you defined the rules, there was a "sem" attribute.
        """
        children = []
        for child in self.children:
            if isinstance(child, str):
                children.append(child)
            else:
                children.append(child.compute_semantics())
        return self.rule.apply_semantics(children)

#========================================= Grammar ==================================================

class Grammar:
    """
    This is where you will define a set of rules to create a grammar.
    """
    def __init__(self, rules, annotators=[], start_symbol=None, induced=False):
        """
        Checks if all of the rules are either lexical or binary and adds them accordingly.

        :param rules: List of Rules
        :param annotators: List of annotators. Do not worry about this for part 1
        """
        self.lexical_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        self.unary_rules = defaultdict(list) # Part 2
        self.annotators = annotators # Part 2
        self.categories = set() # Part 2
        self.start_symbol = start_symbol
        self.max_cell_capacity_hits = 0
        for rule in rules:
            self.add_rule(rule)

        # Part 2: Inducing the rules of the grammar based on the lexicon
        if induced:
            # Make sure that the file path to the provided lexicon file
            lexicon = self.create_lexicon("hw2_lexicon.txt")
            self.induce_grammar_rules(lexicon)

        print('Created grammar with {} rules.'.format(len(rules)))

    def add_rule(self, rule):
        """
        Adds the given rule to tha appropriate group.

        :param rule: The Rule to add
        """
        if rule.is_lexical():
            self.lexical_rules[rule.rhs].append(rule)
        elif rule.is_binary():
            self.binary_rules[rule.rhs].append(rule)
        # Beyond this Part 2
        elif rule.is_unary():
            self.unary_rules[rule.rhs].append(rule)
        elif all([rule.is_cat(rhsi) for rhsi in rule.rhs]):
            self.add_n_ary_rule(rule)
        elif rule.contains_optionals():
            self.add_rule_containing_optional(rule)
        else:
            raise(Exception('Cannot accept rule: {}'.format(rule)))

    def parse_input(self, input_sent):
        """
        Returns a list of parses for the given input. Note that we will do this with a variant
        CYK algorithm, a chart parsing algorithm. This algorithm relies on a chart with a cell
        for every possible span of the input. A span is defined by the starting and ending indices
        where each index defines a token of the input. Like list indexing, we include the starting
        index while excluding the ending index token. For example,

        "one plus one" -> ["one", "plus", "one"]
        span(0,2) -> "one plus"
        span(1,3) -> "plus two"
        span(1,2) -> "plus"

        For a detailed explanation of the pseudocode, visit the Wikipedia page for this algorithm.
        Generally, the algorithm is the following:

        1. Split the input into tokens
        2. Create a chart mapping every span to a possible parse
        3. Iterate over all spans from shorter to longer spans
        4. For every span, check if it is allowed by our grammar, and if so, add it to the chart

        Hint: apply binary rules before unary rules.

        :param input_sent: This is the input sentence
        :return: A chart with parses of all spans. Note that the parses are represented by the Parse class.
        """
        tokens = input_sent.split()
        chart = defaultdict(list)
        n = len(tokens)

        #Apply annotators
        for i, j in product(range(n+1), repeat=2):
            self.apply_annotators(chart, tokens, i, j)

        #Lexical rules
        for i, j in product(range(n+1), repeat=2):
            self.apply_lexical_rules(chart, tokens, i, j)


        for i, j in product(range(n+1), repeat=2):
            self.apply_unary_rules(chart, i, j)

        #Go over all spans, incrementing span length
        for span_length in range(2, n + 1):
            for span_start in range(0, n - span_length + 1):
                i, j = span_start, span_start + span_length
                self.apply_binary_rules(chart, i, j)
                self.apply_unary_rules(chart, i, j)

        parses = chart[(0, len(tokens))]
        if self.start_symbol:
            parses = [parse for parse in parses if parse.rule.lhs == self.start_symbol]
        return parses

    def apply_lexical_rules(self, chart, tokens, i, j):
        """
        TODO: Add the lexical rules to the chart.
        :param chart: chart datastructure
        :param tokens: tokens of input
        :param i: start of span
        :param j: end of span
        """
        for rule in self.lexical_rules[tuple(tokens[i:j])]:
            parse = Parse(rule, tokens[i:j])
            if parse not in chart[(i,j)]:
                chart[(i,j)].append(parse)

    def apply_binary_rules(self, chart, i, j):
        """
        TODO: Add parses to span (i, j) in chart by applying binary rules from grammar.
        :param chart: chart datastructure
        :param i: start of span
        :param j: end of span
        """
        span_length = j - i
        for span_partition in range(1, span_length):
            a = (i, i + span_partition)
            b = (i + span_partition, i + span_length)
            for r in chart[a]:
                for s in chart[b]:
                    for rule in self.binary_rules[(r.rule.lhs, s.rule.lhs)]:
                        if self.check_capacity(chart, i, j):
                            parse = Parse(rule, [r, s])
                            if parse not in chart[(i, j)]:
                                chart[(i, j)].append(parse)

    ############## Part 2 ##############

    def apply_unary_rules(self, chart, i, j):
        """
        TODO: Add parses to chart cell (i, j) by applying unary rules.
        :param chart: chart datastructure
        :param i: start of span
        :param j: end of span

        Hint: 1) iterate over all parses in a cell, (i,j)
              2) for each unary rule in each parse, check capacity and then append rule
        """
        idx = 0
        while idx < len(chart[(i, j)]):
            parse = chart[(i, j)][idx]
            idx += 1
            for rule in self.unary_rules[(parse.rule.lhs,)]:
                if self.check_capacity(chart, i, j):
                    unary_parse = Parse(rule, [parse])
                    if unary_parse not in chart[(i, j)]:
                        chart[(i, j)].append(unary_parse)

    def apply_annotators(self, chart, tokens, i, j):
        """
        TODO: Add parses to chart cell (i, j) by applying annotators. Do not worry about this in part 1.
        :param chart: chart datastructure
        :param tokens: tokens of input
        :param i: start of span
        :param j: end of span

        Hint: 1) If there are annotators
              2) for every annotator, for every token annotation
              3) check capacity and add Rule
        """
        for annotator in self.annotators:
            for annotation in annotator.annotate(tokens[i:j]):
                if self.check_capacity(chart, i, j):
                    rule = Rule(annotation[0], annotation[1])
                    chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def add_n_ary_rule(self, rule):
        """
        Handles adding a rule with three or more non-terminals on the RHS.
        We introduce a new category which covers all elements on the RHS except
        the first, and then generate two variants of the rule: one which
        consumes those elements to produce the new category, and another which
        combines the new category which the first element to produce the
        original LHS category.  We add these variants in place of the
        original rule.  (If the new rules still contain more than two elements
        on the RHS, we'll wind up recursing.)
        For example, if the original rule is:
            Rule('$Z', '$A $B $C $D')
        then we create a new category '$Z_$A' (roughly, "$Z missing $A to the left"),
        and add these rules instead:
            Rule('$Z_$A', '$B $C $D')
            Rule('$Z', '$A $Z_$A')
        Notice that this is automating the currying that we did for part 1 by create EBO
        :param rule: Rule in question
        """
        def add_category(base_name):
            assert rule.is_cat(base_name)
            name = base_name
            while name in self.categories:
                name = name + '_'
            self.categories.add(name)
            return name
        category = add_category('%s_%s' % (rule.lhs, rule.rhs[0]))
        self.add_rule(Rule(category, rule.rhs[1:], lambda sems: sems))
        self.add_rule(Rule(rule.lhs, (rule.rhs[0], category), lambda sems: rule.apply_semantics([sems[0]] + sems[1])))

    def add_rule_containing_optional(self, rule):
        """
        Handles adding a rule which contains an optional element on the RHS.
        We find the leftmost optional element on the RHS, and then generate
        two variants of the rule: one in which that element is required, and
        one in which it is removed.  We add these variants in place of the
        original rule.  (If there are more optional elements further to the
        right, we'll wind up recursing.)
        For example, if the original rule is:
            Rule('$Z', '$A ?$B ?$C $D')
        then we add these rules instead:
            Rule('$Z', '$A $B ?$C $D')
            Rule('$Z', '$A ?$C $D')
        :param rule: Rule in question
        (this is implemented for you)
        """
        # Step 1: Find index of the first optional element on the RHS.
        first = next((idx for idx, elt in enumerate(rule.rhs) if rule.is_optional(elt)), -1)

        # Step 2: get prefix/suffix with the separation being the index found before
        prefix = rule.rhs[:first]
        suffix = rule.rhs[(first + 1):]

        # Step 3: First optional variant - the first optional element gets deoptionalized.
        deoptionalized = (rule.rhs[first][1:],)
        self.add_rule(Rule(rule.lhs, prefix + deoptionalized + suffix, rule.sem))
        # Step 4: Second optional variant - the first optional element gets removed.
        # If the semantics is a value, just keep it as is.
        sem = rule.sem
        # But if it's a function, we need to supply a dummy argument for the removed element.
        if isinstance(rule.sem, FunctionType):
            sem = lambda sems: rule.sem(sems[:first] + [None] + sems[first:])
        # Step 5: Add rule
        self.add_rule(Rule(rule.lhs, prefix + suffix, sem))

    def check_capacity(self, chart, i, j):
        """
        Makes sure that we are not stuck in unary loops for our parses and creates an upper bound
        for our parses. Current it is set at 1000 parses per cell.
        """
        if len(chart[(i, j)]) >= MAX_CELL_CAPACITY:
            self.max_cell_capacity_hits += 1
            lg_max_cell_capacity_hits = math.log(self.max_cell_capacity_hits, 2)
            if int(lg_max_cell_capacity_hits) == lg_max_cell_capacity_hits:
                print('Max cell capacity %d has been hit %d times' % (MAX_CELL_CAPACITY, self.max_cell_capacity_hits))
            return False
        return True

    def create_lexicon(self, lexicon_path):
        """
        This takes in a text file and returns a list of tuples that have the (query, semantic parse, semantic attatchments)
        :param lexicon_path: String - file path to the lexicon
        :return: List of tuples
        """
        lexicon = []
        with open(lexicon_path, "r+") as f:
            lines = [line.split("|") for line in f]
            for line in lines:
                if len(line) == 1:
                    print(line)
                lexicon.append((line[0], line[1], line[2]))
        return lexicon

    def induce_grammar_rules(self, lexicon):
        """
        Given a lexicon created by create_lexicon() which will return a list of tuples, construct a function
        that will induce the lexical rules for our grammar. nltk.tree.Tree.productions() will be very useful
        in parsing both the semantic parse as well as the semantic attachements.
        :param lexicon: List of tuples

        Hint: 1) For every element in lexicon, create a for the semantic parse and the semantics with Tree.fromstring()
              2) For every production from the semantic parse tree you get from Tree.productions(), check if it is lexical
              3) If it is, then go through every semantic attachment from the semantics tree and add the rule to the rule_set
              4) IMPORTANT: do not add rules if rule.lhs = "$Superlative"
              5) Now given the rule set, please only add the top five occuring rules for all left hand sides

        """
        rules_set = {} # map lhs -> dictionary of rule -> count
        for element in lexicon:
            parse = nltk.tree.Tree.fromstring(element[1])
            sems = nltk.tree.Tree.fromstring(element[2])
            for production in parse.productions():
                if self.production_is_lexical(production):
                    for sem_production in sems.productions():
                        sem_items = self.production_to_semantics(sem_production)
                        for sem in sem_items:
                            rule = self.production_to_rule(production, sem)
                            if not rule.lhs == '$Superlative':
                                if rule.rhs not in rules_set:
                                    rules_set[rule.rhs] = {}
                                if rule not in rules_set[rule.rhs]:
                                    rules_set[rule.rhs][rule] = 0
                                rules_set[rule.rhs][rule] += 1
        for rhs, rule_count_dict in rules_set.items():
            most_frequent = list(dict(sorted(rule_count_dict.items(), key=lambda x: x[1], reverse=True)).keys())[:5]
            for rule in most_frequent:
                self.add_rule(rule)

    def production_is_lexical(self, production):
        """
        Checks if an NLTK Production is lexical
        :param production: NLTK Production
        :return: Boolean is lexical or not
        """
        return production.is_lexical() and len(production.rhs()) == 1

    def production_to_rule(self, production, semantics):
        """
        Converts an NLTK Production to our Rule class. Only called for lexical rules.
        If the rule is not lexical, returns none
        production: NLTK.Production
        semantics: semantic attachment from lexicon
        returns: Rule or None
        """
        lhs = production.lhs().symbol().replace(',','')
        rhs = ' '.join([elements.symbol() if isinstance(elements, nltk.grammar.Nonterminal) else elements for elements in production.rhs()])
        rhs = rhs.replace(',','')
        return Rule(lhs, rhs, semantics)

    def production_to_semantics(self, production):
        """
        Converts an NLTK Productiona to a semantic attachment.
        :param production: NLTK Production
        :return: String
        """
        semantics = [production.lhs().symbol().replace(',','')]
        semantics += [elements.symbol().replace(',','') if isinstance(elements, nltk.grammar.Nonterminal) else elements.replace(',','') for elements in production.rhs()]
        return semantics

    def print_chart(self, chart):
        """
        Print the chart.  Useful for debugging.
        """
        spans = sorted(list(chart.keys()), key=(lambda span: span[0]))
        spans = sorted(spans, key=(lambda span: span[1] - span[0]))
        for span in spans:
            if len(chart[span]) > 0:
                print('%-12s' % str(span), end=' ')
                print(chart[span][0])
                for entry in chart[span][1:]:
                    print('%-12s' % ' ', entry)

#========================================= Part 1 ==================================================

def define_arithmetic_rules():
    """ TODO
    This is where you can define all of the rules for our arithmetic lexicon. Note that our examples range
    from 1 to 4 while all of the operations are +, -, *, and ~ (negation). Note that negation and subtraction
    are both represented as "minus".
    """
    numeral_rules = [
        Rule('$Expr', 'one', Number(1)),
        Rule('$Expr', 'two', Number(2)),
        Rule('$Expr', 'three', Number(3)),
        Rule('$Expr', 'four', Number(4)),
        Rule('$Expr', 'five', Number(5)),
        Rule('$Expr', 'six', Number(6)),
        Rule('$Expr', 'seven', Number(7)),
        Rule('$Expr', 'eight', Number(8)),
        Rule('$Expr', 'nine', Number(9))
    ]

    operator_rules = [
        Rule('$BinOp', 'plus', '+'),
        Rule('$BinOp', 'minus', '-'),
        Rule('$BinOp', 'times', '*'),
        Rule('$UnOp', 'minus', '~')
    ]

    # Remember that we need these to be in chomsky normal form
    compositional_rules = [
        Rule('$EBO', '$Expr $BinOp', lambda sem: tuple(sem)),
        Rule('$Expr', '$EBO $Expr', lambda sem: Op2(sem[0][1], sem[0][0], sem[1])),
        Rule('$Expr', '$UnOp $Expr', lambda sem: Op1(sem[0], sem[1]))
    ]

    return numeral_rules + operator_rules + compositional_rules

def get_parses(examples, grammar):
    """
    This function will compute the parses defined by the chart constructed through the CYK algorithm
    and print all potential parses.

    :param examples: List of input sentences to parse
    :param grammar: The grammar rules to evaluate our inputs
    :return: A list of parses for each example
    """
    all_parses = []
    for example in examples:
        parses = grammar.parse_input(example)
        all_parses.append(parses)
        print()
        print('{:16} {}'.format('input', example))
        for idx, parse in enumerate(parses):
            print('{:16} {}'.format('parse {}'.format(idx), parse))
    return all_parses

def get_semantics(all_parses):
    """
    This function prints the parses with the attached semantics as well as the evaluated values of the input.
    """
    for parses in all_parses:
        for parse in parses:
            print()
            print(parse.semantics)
            print(parse.semantics.eval())

if __name__ == '__main__':
    examples = [
        "one plus one",
        "one plus two",
        "one plus three",
        "two plus two",
        "two plus three",
        "three plus one",
        "three plus minus two",
        "two plus two",
        "three minus two",
        "minus three minus two",
        "two times two",
        "two times three",
        "three plus three minus two",
        "minus three",
        "three plus two",
        "two times two plus three",
        "minus four"
    ]

    # Step 1: Define all of the rules
    arithmetic_rules = define_arithmetic_rules()

    # Step 2: Construct the Grammar! Hint: In our solution, we have 11 rules
    arithmetic_grammar = Grammar(arithmetic_rules)

    # Step 3.1: Implement the CYK algorithm and apply them to the examples
    parses = get_parses(examples, arithmetic_grammar)

    # Step 3.2: Now add semantics to our parsed examples. Note that this is the compute_semantics of the Parse class.
    get_semantics(parses)
