from model import Model
from metrics import denotation_match_metrics, DenotationAccuracyMetric
from collections import defaultdict
from operator import itemgetter
from geobase import GeobaseReader
from graph_kb import GraphKB, GraphKBExecutor
from geo880 import geo880_train_examples, geo880_test_examples
from grammar import *
from experiment import sample_wins_and_losses

class GeoQueryDomain():
    """
    This class represents our geography query class.
    """
    def __init__(self):
        """
        As the handout details, we have provided the stencil code for the data structure to access the data
        as well as the executor to evaluate a parse on this dataset (i.e. get the answer from a query). If you
        would like to see the actual data. Please take a look at geo880.py.
        """
        self.geobase = GraphKB(GeobaseReader().tuples)
        self.geobase_executor = GraphKBExecutor(self.geobase)
        self.train_examples = geo880_train_examples
        self.test_examples = geo880_test_examples

    def rules(self, with_manual=True):
        """
        The ability to toggle whether or not we include the manual lexical rules. 
        :param with_manual: Boolean to toggle the sets of rules.
        """
        if with_manual:
            return self.manual_rules()
        else:
            return self.nonlexical_rules()


    def nonlexical_rules(self):
        """
        These are all of the nonlexical rules that were manually defined.
        """
        # These are the words that we will define as optional
        optional_words = ['the', 'what', 'is', 'in', 'of', 'how', 'many', 'are', 'which', 'that',
                          'with', 'has', 'major', 'does', 'have', 'where', 'me', 'there', 'give',
                          'name', 'all', 'a', 'by', 'you', 'to', 'tell', 'other', 'it', 'do',
                          'whose', 'show', 'one', 'on', 'for', 'can', 'whats', 'urban', 'them',
                          'list', 'exist', 'each', 'could', 'about', '.', '?']

        rules_optional = [
            Rule('$ROOT', '?$Optionals $Query ?$Optionals', lambda sem: sem[1]),
            Rule('$Optionals', '$Optional ?$Optionals'),
        ] + [Rule('$Optional', word) for word in optional_words]

        rules_collection_entity = [
            Rule('$Query', '$Collection', lambda sem: sem[0]),
            Rule('$Collection', '$Entity', lambda sem: sem[0])
        ]
        # First let us define a $Type as a $Collection
        rules_type = [
            Rule('$Collection', '$Type', lambda sems: sems[0])
        ]

        # These are the compositional rules for Relations and Joins
        rules_relations = [
            Rule('$Collection', '$Relation ?$Optionals $Collection', lambda sems: sems[0](sems[2])),
            Rule('$Relation', '$FwdRelation', lambda sems: (lambda arg: (sems[0], arg))),
            Rule('$Relation', '$RevRelation', lambda sems: (lambda arg: (arg, sems[0]))),

            Rule('$FwdRelation', '$FwdBordersRelation', 'borders'),
            Rule('$FwdRelation', '$FwdTraversesRelation', 'traverses'),
            Rule('$RevRelation', '$RevTraversesRelation', 'traverses'),
            Rule('$FwdRelation', '$FwdContainsRelation', 'contains'),
            Rule('$RevRelation', '$RevContainsRelation', 'contains'),
            Rule('$RevRelation', '$RevCapitalRelation', 'capital'),
            Rule('$RevRelation', '$RevLowestPointRelation', 'lowest_point'),
            Rule('$RevRelation', '$RevHighestElevationRelation', 'highest_elevation'),
            Rule('$RevRelation', '$RevHeightRelation', 'height'),
            Rule('$RevRelation', '$RevAreaRelation', 'area'),
            Rule('$RevRelation', '$RevPopulationRelation', 'population'),
            Rule('$RevRelation', '$RevLengthRelation', 'length')
        ]

        # Rules for intersection
        rules_intersection = [
            Rule('$Collection', '$Collection $Collection', lambda sems: ('.and', sems[0], sems[1])),
            Rule('$Collection', '$Collection $Optional $Collection', lambda sems: ('.and', sems[0], sems[2])),
            Rule('$Collection', '$Collection $Optional $Optional $Collection', lambda sems: ('.and', sems[0], sems[3]))
        ]

        # Rules for Superlatives
        rules_superlatives = [
            Rule('$Collection', '$Superlative ?$Optionals $Collection', lambda sems: sems[0] + (sems[2],)),
            Rule('$Collection', '$Collection ?$Optionals $Superlative', lambda sems: sems[2] + (sems[0],)),

            Rule('$Superlative', 'largest', ('.argmax', 'area')),
            Rule('$Superlative', 'largest', ('.argmax', 'population')),
            Rule('$Superlative', 'biggest', ('.argmax', 'area')),
            Rule('$Superlative', 'biggest', ('.argmax', 'population')),
            Rule('$Superlative', 'smallest', ('.argmin', 'area')),

            Rule('$Superlative', '$MostLeast $RevRelation', lambda sems: (sems[0], sems[1])),
            Rule('$MostLeast', 'most', '.argmax'), 
            Rule('$MostLeast', 'least', '.argmin')
        ]

        # The reverse rule
        rules_reverse_joins = [
            Rule('$Collection', '$Collection ?$Optionals $Relation', lambda sems: self.reverse(sems[2])(sems[0]))
        ]

        return rules_optional + rules_collection_entity + rules_type + rules_relations + rules_intersection + rules_superlatives + rules_reverse_joins


    def manual_rules(self):
        """
        These rules were manually constructed.
        """
        # These are the words that we will define as optional
        optional_words = ['the', 'what', 'is', 'in', 'of', 'how', 'many', 'are', 'which', 'that',
                         'with', 'has', 'major', 'does', 'have', 'where', 'me', 'there', 'give',
                         'name', 'all', 'a', 'by', 'you', 'to', 'tell', 'other', 'it', 'do',
                         'whose', 'show', 'one', 'on', 'for', 'can', 'whats', 'urban', 'them',
                         'list', 'exist', 'each', 'could', 'about', '.', '?']

        rules_optional = [
            Rule('$ROOT', '?$Optionals $Query ?$Optionals', lambda sem: sem[1]),
            Rule('$Optionals', '$Optional ?$Optionals'),
        ] + [Rule('$Optional', word) for word in optional_words]

        rules_collection_entity = [
            Rule('$Query', '$Collection', lambda sem: sem[0]),
            Rule('$Collection', '$Entity', lambda sem: sem[0])
        ]
        # Now we will create the lexical rules for Types. First let us define a $Type as a $Collection
        rules_type = [
            Rule('$Collection', '$Type', lambda sems: sems[0]),
            # Now here are a few examples
            Rule('$Type', 'state', 'state'),
            Rule('$Type', 'states', 'state'),
            Rule('$Type', 'city', 'city'),
            Rule('$Type', 'cities', 'city'),
            Rule('$Type', 'big cities', 'city'),
            Rule('$Type', 'towns', 'city'),
            Rule('$Type', 'river', 'river'),
            Rule('$Type', 'rivers', 'river'),
            Rule('$Type', 'mountain', 'mountain'),
            Rule('$Type', 'mountains', 'mountain'),
            Rule('$Type', 'mount', 'mountain'),
            Rule('$Type', 'peak', 'mountain'),
            Rule('$Type', 'road', 'road'),
            Rule('$Type', 'roads', 'road'),
            Rule('$Type', 'lake', 'lake'),
            Rule('$Type', 'lakes', 'lake'),
            Rule('$Type', 'country', 'country'),
            Rule('$Type', 'countries', 'country')
        ]

        # These are the rules for Relations and Joins
        rules_relations = [
            Rule('$Collection', '$Relation ?$Optionals $Collection', lambda sems: sems[0](sems[2])),
            Rule('$Relation', '$FwdRelation', lambda sems: (lambda arg: (sems[0], arg))),
            Rule('$Relation', '$RevRelation', lambda sems: (lambda arg: (arg, sems[0]))),

            # These all describe the forward relationship for borders
            Rule('$FwdRelation', '$FwdBordersRelation', 'borders'),
            # Here are some examples, please come up with 5 more examples of $FwdBordersRelation
            Rule('$FwdBordersRelation', 'border'),
            Rule('$FwdBordersRelation', 'bordering'),
            Rule('$FwdBordersRelation', 'borders'),
            Rule('$FwdBordersRelation', 'neighbor'),
            Rule('$FwdBordersRelation', 'neighboring'),
            Rule('$FwdBordersRelation', 'surrounding'),
            Rule('$FwdBordersRelation', 'next to'),

            # These all describe the forward relationship for traverses
            Rule('$FwdRelation', '$FwdTraversesRelation', 'traverses'),
            # Here are some examples, please come up with 10 more examples of $FwdBordersRelation
            Rule('$FwdTraversesRelation', 'cross over'),
            Rule('$FwdTraversesRelation', 'flow through'),
            Rule('$FwdTraversesRelation', 'flowing through'),
            Rule('$FwdTraversesRelation', 'flows through'),
            Rule('$FwdTraversesRelation', 'go through'),
            Rule('$FwdTraversesRelation', 'goes through'),
            Rule('$FwdTraversesRelation', 'in'),
            Rule('$FwdTraversesRelation', 'pass through'),
            Rule('$FwdTraversesRelation', 'passes through'),
            Rule('$FwdTraversesRelation', 'run through'),
            Rule('$FwdTraversesRelation', 'running through'),
            Rule('$FwdTraversesRelation', 'runs through'),
            Rule('$FwdTraversesRelation', 'traverse'),
            Rule('$FwdTraversesRelation', 'traverses'),

            # Now for the reverse relation for traverses
            Rule('$RevRelation', '$RevTraversesRelation', 'traverses'),
            Rule('$RevTraversesRelation', 'has'),
            Rule('$RevTraversesRelation', 'have'),  # example: 'how many states have major rivers'
            Rule('$RevTraversesRelation', 'lie on'),
            Rule('$RevTraversesRelation', 'next to'),
            Rule('$RevTraversesRelation', 'traversed by'),
            Rule('$RevTraversesRelation', 'washed by'),

            # Forward relation for contains
            Rule('$FwdRelation', '$FwdContainsRelation', 'contains'),
            Rule('$FwdContainsRelation', 'have'), # example: 'how many states have a city named springfield'
            Rule('$FwdContainsRelation', 'has'),

            # Reverse relation for contains
            Rule('$RevRelation', '$RevContainsRelation', 'contains'),
            Rule('$RevContainsRelation', 'contained by'),
            Rule('$RevContainsRelation', 'in'),
            Rule('$RevContainsRelation', 'found in'),
            Rule('$RevContainsRelation', 'located in'),
            Rule('$RevContainsRelation', 'of'),

            # Reverse relation of capital
            Rule('$RevRelation', '$RevCapitalRelation', 'capital'),
            Rule('$RevCapitalRelation', 'capital'),
            Rule('$RevCapitalRelation', 'capitals'),

            Rule('$RevRelation', '$RevHighestPointRelation', 'highest_point'),
            Rule('$RevHighestPointRelation', 'high point'),
            Rule('$RevHighestPointRelation', 'high points'),
            Rule('$RevHighestPointRelation', 'highest point'),
            Rule('$RevHighestPointRelation', 'highest points'),

            Rule('$RevRelation', '$RevLowestPointRelation', 'lowest_point'),
            Rule('$RevLowestPointRelation', 'low point'),
            Rule('$RevLowestPointRelation', 'low points'),
            Rule('$RevLowestPointRelation', 'lowest point'),
            Rule('$RevLowestPointRelation', 'lowest points'),
            Rule('$RevLowestPointRelation', 'lowest spot'),

            Rule('$RevRelation', '$RevHighestElevationRelation', 'highest_elevation'),
            Rule('$RevHighestElevationRelation', 'highest elevation'), # why the ?

            Rule('$RevRelation', '$RevHeightRelation', 'height'),
            Rule('$RevHeightRelation', 'elevation'),
            Rule('$RevHeightRelation', 'height'),
            Rule('$RevHeightRelation', 'high'),
            Rule('$RevHeightRelation', 'tall'),

            Rule('$RevRelation', '$RevAreaRelation', 'area'),
            Rule('$RevAreaRelation', 'area'),
            Rule('$RevAreaRelation', 'big'),
            Rule('$RevAreaRelation', 'large'),
            Rule('$RevAreaRelation', 'size'),

            Rule('$RevRelation', '$RevPopulationRelation', 'population'),
            Rule('$RevPopulationRelation', 'big'),
            Rule('$RevPopulationRelation', 'large'),
            Rule('$RevPopulationRelation', 'populated'),
            Rule('$RevPopulationRelation', 'population'),
            Rule('$RevPopulationRelation', 'populations'),
            Rule('$RevPopulationRelation', 'populous'),
            Rule('$RevPopulationRelation', 'size'),

            Rule('$RevRelation', '$RevLengthRelation', 'length'),
            Rule('$RevLengthRelation', 'length'),
            Rule('$RevLengthRelation', 'long'),
        ]

        # Rules for intersection
        rules_intersection = [
            Rule('$Collection', '$Collection $Collection', lambda sems: ('.and', sems[0], sems[1])),
            Rule('$Collection', '$Collection $Optional $Collection', lambda sems: ('.and', sems[0], sems[2])),
            Rule('$Collection', '$Collection $Optional $Optional $Collection', lambda sems: ('.and', sems[0], sems[3]))
        ]

        # Rules for Superlatives
        rules_superlatives = [
            Rule('$Collection', '$Superlative ?$Optionals $Collection', lambda sems: sems[0] + (sems[2],)),
            Rule('$Collection', '$Collection ?$Optionals $Superlative', lambda sems: sems[2] + (sems[0],)),

            Rule('$Superlative', 'largest', ('.argmax', 'area')),
            Rule('$Superlative', 'largest', ('.argmax', 'population')),
            Rule('$Superlative', 'biggest', ('.argmax', 'area')),
            Rule('$Superlative', 'biggest', ('.argmax', 'population')),
            Rule('$Superlative', 'smallest', ('.argmin', 'area')),
            Rule('$Superlative', 'smallest', ('.argmin', 'population')),
            Rule('$Superlative', 'longest', ('.argmax', 'length')),
            Rule('$Superlative', 'shortest', ('.argmin', 'length')),
            Rule('$Superlative', 'tallest', ('.argmax', 'height')),
            Rule('$Superlative', 'highest', ('.argmax', 'height')),

            Rule('$Superlative', '$MostLeast $RevRelation', lambda sems: (sems[0], sems[1])),
            Rule('$MostLeast', 'most', '.argmax'), 
            Rule('$MostLeast', 'least', '.argmin'),
            Rule('$MostLeast', 'lowest', '.argmin'),
            Rule('$MostLeast', 'greatest', '.argmax'),
            Rule('$MostLeast', 'highest', '.argmax')
        ]

        # The reverse rule
        rules_reverse_joins = [
            Rule('$Collection', '$Collection ?$Optionals $Relation', lambda sems: self.reverse(sems[2])(sems[0]))
        ]

        return rules_optional + rules_collection_entity + rules_type + rules_relations + rules_intersection + rules_superlatives + rules_reverse_joins

    def reverse(self, relation_sem):
        """
        relation_sem is a lambda function which takes an arg and forms a pair, either (rel, arg) or (arg, rel).
        Return a function that applies relation_sem to its arguments, and returns a tuple that
          swaps the order of the pair
        """
        def apply_and_swap(arg):
            pair = relation_sem(arg)
            return (pair[1], pair[0])
        return apply_and_swap

    def annotators(self):
        """
        This function will return a list of annotators that are relevant to adding the semantics of our Rules.
        """
        return [TokenAnnotator(), GeobaseAnnotator(self.geobase)]

    def model(self):
        """
        This is a helper function to fun the evaluations
        """
        return Model(grammar=self.grammar(), executor=self.execute)

    def grammar(self):
        """
        This funtion creates the grammar given the rules that you defined earlier.
        """
        # Note that if you want to change which sets of rules to use you should toggle the with_manual inside self.rules()
        return Grammar(rules=self.rules(with_manual=True), annotators=self.annotators(), start_symbol='$ROOT', induced=True)

    def execute(self, semantics):
        """
        This will allow the executor to determine the answers of the geobase queries
        """
        return self.geobase_executor.execute(semantics)

    def metrics(self):
        """
        The metrics that we are using for our experiments.
        """
        return denotation_match_metrics()

    def training_metric(self):
        """
        Similar to metrics.
        """
        return DenotationAccuracyMetric()


class TokenAnnotator:
    """
    This is an example of an annotator for reference.
    """
    def annotate(self, tokens):
        if len(tokens) == 1:
            return [('$Token', tokens[0])]
        else:
            return []

class GeobaseAnnotator:
    """
    Creates lexical rules for $Entity using the Geobase knowledge base
    """
    def __init__(self, geobase):
        self.geobase = geobase

    def annotate(self, tokens):
        """
        TODO: Create an annotator will annotate the $Entity category with the name
        :param tokens: List of tokens to make up query
        :return: list of tuples of the form ('$Entity', place)

        Hint: Note that the GraphKB datastructure allows for reverse queries by calling binaries_rev['name'][query]
        which will return a list of names for the given query.
        """
        raise NotImplementedError("GeobaseAnnotator.annotate")

if __name__ == '__main__':
    domain = GeoQueryDomain()
    # Note that you can keep running this script to evaluate your rules. 
    # Hint: First complete the annotator and the grammar class before running these tests for part 2.
    sample_wins_and_losses(domain=domain)
