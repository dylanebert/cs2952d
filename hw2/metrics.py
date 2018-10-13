# An evaluation metric is a function that takes a list of parses and an example,
# and returns a number.
class Metric:
    def name(self):
        return ''

    def evaluate(self, example, parses):
        return 0.0


class DenotationAccuracyMetric(Metric):
    def name(self):
        return 'denotation accuracy'

    def evaluate(self, example, parses):
        return 1.0 if parses and parses[0].denotation == example.denotation else 0.0


class DenotationOracleAccuracyMetric(Metric):
    def name(self):
        return 'denotation oracle accuracy'

    def evaluate(self, example, parses):
        for parse in parses:
            if parse.denotation == example.denotation:
                return 1.0
        return 0.0


class NumParsesMetric(Metric):
    def name(self):
        return 'number of parses'

    def evaluate(self, example, parses):
        return len(parses)


class SpuriousAmbiguityMetric(Metric):
    """
    Returns a value on [0, 1] which reflects the degree of spurious ambiguity.
    Returns 0.0 if each parse has unique semantics.
    Returns 1.0 if there are multiple parses, all sharing the same semantics.
    In general, returns a value which can be interpreted as the fraction of
    parses whose semantics were already produced by another parse.
    """

    def name(self):
        return 'spurious ambiguity'

    def evaluate(self, example, parses):
        if len(parses) == 1:
            return 0.0
        sems = set([str(parse.semantics) for parse in parses])
        # This conditional should be redundant with the final line.
        # But without it, we can return -0.0, which looks weird.
        if len(sems) == len(parses):
            return 0.0
        return 1.0 * (len(parses) - len(sems)) / (len(parses) - 1)


def denotation_match_metrics():
    return [
        DenotationAccuracyMetric(),
        DenotationOracleAccuracyMetric(),
        NumParsesMetric(),
        SpuriousAmbiguityMetric(),
    ]
