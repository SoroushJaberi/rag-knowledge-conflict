"""Unit tests for scoring and metrics (no GPU needed): python test_scoring.py"""
import math

from metrics import auroc, bootstrap_ci, ece
from scoring import classify, confidence, is_abstain, matches, normalize, split_answer


def test_normalize_and_match():
    assert normalize("The Beatles!") == "beatles"
    assert normalize("Three episodes") == "3 episodes"
    assert matches("It was Bryan Cranston.", ["Bryan Cranston"])
    assert matches("Cranston", ["Bryan Cranston"])          # short answer inside the gold span
    assert not matches("1", ["1990"])                         # token boundaries, not substrings
    assert not matches("Tom", ["Tom Hanks"])                  # under 4 characters: too short to count
    assert matches("23", ["23"]) and matches("twenty", ["20"])
    assert not matches("9.21", ["nine"]) and matches("820,445", ["820445"])  # decimals/thousands stay one number
    assert matches("2015, 2016", ["2016"])


def test_classify():
    org, sub = ["Bryan Cranston"], ["Christopher Meloni"]
    assert classify("Bryan Cranston", org, sub) == "original"
    assert classify("Christopher Meloni", org, sub) == "substituted"
    assert classify("Bryan Cranston or Christopher Meloni", org, sub) == "both"
    assert classify("I don't know.", org, sub) == "abstain"
    assert classify("Tom Hanks", org, sub) == "other"
    assert classify(None, org, sub) == "truncated"
    assert is_abstain("I do not know") and not is_abstain("Senegal")


def test_split_answer():
    vocab = {1: "</think>", 2: "<eos>", 10: "\n", 11: "Sen", 12: "egal", 20: "hmm"}
    dec = lambda ids: "".join(vocab[i] for i in ids)
    ids, lps = [20, 20, 1, 10, 11, 12, 2], [-1, -1, -0.1, -0.01, -0.5, -0.1, -0.01]
    reason, ans, alps, trunc = split_answer(ids, lps, 1, {1, 2}, True, dec)
    assert reason == [20, 20] and ans == [10, 11, 12] and alps == [-0.5, -0.1] and not trunc
    vocab.update({30: "Bryan", 31: " ", 32: "Cranston"})                       # spaces inside the answer survive
    assert dec(split_answer([30, 31, 32, 2], [-0.1, 0, -0.2, 0], 1, {1, 2}, False, dec)[1]) == "Bryan Cranston"
    assert split_answer([20, 20], [-1, -1], 1, {1, 2}, True, dec)[3]          # thinking never closed -> truncated
    assert split_answer([11, 12, 2], [-0.2, -0.2, 0], 1, {1, 2}, False, dec)[1] == [11, 12]
    assert math.isclose(confidence([-0.5, -0.1]), math.exp(-0.3))
    assert confidence([]) is None


def test_metrics():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(auroc([0.5], [1]))
    assert ece([1.0, 1.0], [1, 1]) == 0.0 and math.isclose(ece([0.9, 0.9], [0, 0]), 0.9)
    lo, hi = bootstrap_ci(lambda x: x.mean(), [0, 1] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
