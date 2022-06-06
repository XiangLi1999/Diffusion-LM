import torch
import spacy, nltk
from nltk.tree import Tree
import numpy as np

def collapse_unary_strip_pos(tree, strip_top=True):
    """Collapse unary chains and strip part of speech tags."""

    def strip_pos(tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return tree[0]
        else:
            return nltk.tree.Tree(tree.label(), [strip_pos(child) for child in tree])

    collapsed_tree = strip_pos(tree)
    collapsed_tree.collapse_unary(collapsePOS=True, joinChar="::")
    if collapsed_tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
        if strip_top:
            if len(collapsed_tree) == 1:
                collapsed_tree = collapsed_tree[0]
            else:
                collapsed_tree.set_label("")
        elif len(collapsed_tree) == 1:
            collapsed_tree[0].set_label(
                collapsed_tree.label() + "::" + collapsed_tree[0].label())
            collapsed_tree = collapsed_tree[0]
    return collapsed_tree

def _get_labeled_spans(tree, spans_out, start):
    if isinstance(tree, str):
        return start + 1

    assert len(tree) > 1 or isinstance(
        tree[0], str
    ), "Must call collapse_unary_strip_pos first"
    end = start
    for child in tree:
        end = _get_labeled_spans(child, spans_out, end)
    # Spans are returned as closed intervals on both ends
    spans_out.append((start, end - 1, tree.label()))
    return end

def get_labeled_spans(tree):
    """Converts a tree into a list of labeled spans.
    Args:
        tree: an nltk.tree.Tree object
    Returns:
        A list of (span_start, span_end, span_label) tuples. The start and end
        indices indicate the first and last words of the span (a closed
        interval). Unary chains are collapsed, so e.g. a (S (VP ...)) will
        result in a single span labeled "S+VP".
    """
    tree = collapse_unary_strip_pos(tree)
    spans_out = []
    _get_labeled_spans(tree, spans_out, start=0)
    return spans_out

def padded_chart_from_spans(label_vocab, spans, ):
    num_words = 64
    chart = np.full((num_words, num_words), -100, dtype=int)
    # chart = np.tril(chart, -1)
    # Now all invalid entries are filled with -100, and valid entries with 0
    for start, end, label in spans:
        if label in label_vocab:
            chart[start, end] = label_vocab[label]
    return chart

def chart_from_tree(label_vocab, tree, verbose=False):
    spans = get_labeled_spans(tree)
    num_words = len(tree.leaves())
    chart = np.full((num_words, num_words), -100, dtype=int)
    chart = np.tril(chart, -1)
    # Now all invalid entries are filled with -100, and valid entries with 0
    # print(tree)
    for start, end, label in spans:
        # Previously unseen unary chains can occur in the dev/test sets.
        # For now, we ignore them and don't mark the corresponding chart
        # entry as a constituent.
        # print(start, end, label)
        if label in label_vocab:
            chart[start, end] = label_vocab[label]
    if not verbose:
        return chart
    else:
        return chart, spans

def pad_charts(charts, padding_value=-100):
    """
    Our input text format contains START and END, but the parse charts doesn't.
    NEED TO: update the charts, so that we include these two, and set their span label to 0.

    :param charts:
    :param padding_value:
    :return:
    """
    max_len = 64
    padded_charts = torch.full(
        (len(charts), max_len, max_len),
        padding_value,
    )
    padded_charts = np.tril(padded_charts, -1)
    # print(padded_charts[-2:], padded_charts.shape)
    # print(padded_charts[1])
    for i, chart in enumerate(charts):
        # print(chart, len(chart), len(chart[0]))
        chart_size = len(chart)
        padded_charts[i, 1:chart_size+1, 1:chart_size+1] = chart

    # print(padded_charts[-2:], padded_charts.shape)
    return padded_charts