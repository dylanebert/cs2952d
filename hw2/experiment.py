from collections import defaultdict
import random
from example import Example
from model import Model


def print_sample_outcomes(model=None, examples=[], name=None, metric=None, metric_test=None, k=10):
    candidates = []
    for example in examples:
        parses = model.parse_input(example.input)
        metric_value = metric.evaluate(example, parses)
        if metric_test(metric_value):
            candidates.append(example)
    k = min(k, len(candidates))
    samples = random.sample(candidates, k)
    print('%d of %d %s on %s:\n' % (k, len(candidates), name, metric.name()))
    inputs = [example.input for example in samples]
    for input in sorted(inputs):
        print(' ', input)
    print()
    return samples


def sample_wins_and_losses(domain=None):
    metric = domain.training_metric()
    model = domain.model()
    train_examples = domain.train_examples
    evaluate_model(model=model,
                   examples=train_examples,
                   metrics=domain.metrics())

def evaluate_model(model=None,
                   examples=[],
                   examples_label=None,
                   metrics=None):
    print('=' * 80)
    print('Evaluating on %d %sexamples\n' % (
        len(examples), examples_label + ' ' if examples_label else ''))
    print('-' * 80)
    metric_values = defaultdict(int)
    for example in examples:
        parses = model.parse_input(example.input)
        for metric in metrics:
            metric_value = metric.evaluate(example, parses)
            metric_values[metric.name()] += metric_value
    print('Over %d examples:' % len(examples))
    print()
    for metric in metrics:
        print('%-34s %.3f' % (metric.name(), 1.0 * metric_values[metric.name()] / len(examples)))
    print()
