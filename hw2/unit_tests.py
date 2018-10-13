from grammar import *
from geobase import GeobaseReader
from graph_kb import GraphKB, GraphKBExecutor
from geography import GeobaseAnnotator

def rule_tests():
    def test_is_cat():
        rule = Rule("$Cat", "terminal")
        assert(rule.is_cat("$Cat"))
        assert(not rule.is_cat("terminal"))

    def test_is_lexical():
        assert(Rule("$LHS", "terminal").is_lexical())
        assert(not Rule("$LHS", "$Cat1 $Cat2").is_lexical())

    def test_is_binary():
        assert(Rule("$LHS", "$Cat1 $Cat2").is_binary())
        assert(not Rule("$LHS", "terminal").is_unary())
        assert(not Rule("$LHS", "$Cat1 $Cat2 $Cat3").is_binary())

    def test_is_unary():
        assert(Rule("$LHS", "$Cat1").is_unary())
        assert(not Rule("$LHS", "terminal").is_unary())

    def test_is_optional():
        rule = Rule("$Cat", "?$Optional $NonOptional")
        assert(rule.is_optional("?$Optional"))
        assert(not rule.is_optional("$NonOptional"))

    def test_contains_optionals():
        assert(Rule("$Cat", "?$Optional $NonOptional").contains_optionals())
        assert(not Rule("$Cat", "$NonOptional $NonOptional").contains_optionals())
        assert(not Rule("$?Optional", "$NonOptional $NonOptional").contains_optionals())

    test_is_cat()
    test_is_lexical()
    test_is_binary()
    test_is_unary()
    test_is_optional()
    test_contains_optionals()

lexical_rules = [
        Rule("$Number", "one", 1),
        Rule("$Number", "two", 2),
        Rule("$Number", "three", 3),
        Rule("$Number", "four", 4),
        Rule("$Number", "five", 5)
    ]
compositional_rules = [
        Rule("$Double", "$Number $Number", lambda sem: sem[0] + sem[1])
    ]
rules = lexical_rules + compositional_rules
# Note that the constructor calls add_rule()
grammar = Grammar(rules)

def parse_tests():
    """
    Unit tests for the Parse class
    """
    global lexical_rules
    global compositional_rules
    def test_compute_semantics():
        lexical_parses = [Parse(rule, []) for rule in lexical_rules]
        # This would be the parse for 2+3
        compositional_parse = Parse(compositional_rules[0], lexical_parses[1:3])
        # Lexical Semantics
        for idx, l_parse in enumerate(lexical_parses):
            assert(l_parse.compute_semantics() == lexical_parses[idx].rule.sem)
        # Non Lexical Semantics
        assert(compositional_parse.compute_semantics() == 5)


    test_compute_semantics()


def simple_grammar_tests():
    """
    Unit tests for the Grammar class for Part 1
    """
    global grammar
    def test_add_rule():
        assert(len(grammar.lexical_rules) == 5)
        assert(len(grammar.binary_rules) == 1)

    def test_apply_lexical_rules():
        input_sent = "one one"
        tokens = input_sent.split()
        chart = defaultdict(list)
        grammar.apply_lexical_rules(chart, tokens, 0, 1)
        grammar.apply_lexical_rules(chart, tokens, 1, 2)
        chart_entry_0_1 = "($Number, one)"
        chart_entry_1_2 = "($Number, one)"
        assert(str(chart[(0,1)][0]) == chart_entry_0_1)
        assert(str(chart[(0,1)][0]) == chart_entry_1_2)

    def test_apply_binary_rules():
        input_sent = "one one"
        tokens = input_sent.split()
        chart = defaultdict(list)
        grammar.apply_lexical_rules(chart, tokens, 0, 1)
        grammar.apply_lexical_rules(chart, tokens, 1, 2)
        grammar.apply_binary_rules(chart, 0, 2)
        chart_entry_0_2 = "($Double, ($Number, one) ($Number, one))"
        assert(str(chart[(0,2)][0]) == chart_entry_0_2)

    def test_parse_input():
        example = "one one"
        parsed_answer = "($Double, ($Number, one) ($Number, one))"
        assert(str(grammar.parse_input(example)[0]) == parsed_answer)

    test_add_rule()
    test_apply_lexical_rules()
    test_apply_binary_rules()
    test_parse_input()

def annotator_tests():
    """
    Unit test for the GeobaseAnnotator
    """
    annotator = GeobaseAnnotator(GraphKB(GeobaseReader().tuples))
    assert(annotator.annotate(["montana"])[0] == ('$Entity', '/state/montana'))



if __name__ == '__main__':
    rule_tests()
    parse_tests()
    simple_grammar_tests()
    annotator_tests()
    print("All tests passed! :)")
