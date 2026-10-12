"""Unit tests for the study v2 code (no GPU needed): python test_v2.py"""
import json
import math
from pathlib import Path

import budget as bf
from metrics import cohen_kappa, sign_test
from plausible import check, clean_candidate, kind, substitute
from run_eval import configs, run_name
from scoring import classify, confidence, planted_answer, split_answer

# A character-level fake model: token id = character code; 1 = </think>, 2 = end of text, 3 = <think>.
THINK_END, EOS = 1, 2
encode = lambda s: [ord(c) for c in s.replace("</think>", "\x01").replace("<think>", "\x03")]
decode = lambda ids: "".join({1: "</think>", 2: "", 3: "<think>"}.get(i, chr(i)) for i in ids)
STOP_RAW = "".join(map(chr, encode(bf.STOP_TEXT)))
ANSWER = "\n\nans\x02"


class FakeModel:
    """Thinks T tokens ("t"), closes, answers "ans". Answers right away after the stop text; thinks 5 more after Wait."""
    eos_ids = {EOS}

    def __init__(self, T=40):
        self.T, self.calls = T, 0

    def rest(self, g):
        if g.endswith(STOP_RAW):
            return "ans\x02"
        if "\x01" in g:
            tail = g.rsplit("\x01", 1)[1]
            assert ANSWER.startswith(tail), repr(g)
            return ANSWER[len(tail):]
        if g.endswith("Wait"):
            return "u" * 5 + "\x01" + ANSWER
        full = "\x03" + "t" * self.T + "\x01" + ANSWER
        assert full.startswith(g), repr(g)
        return full[len(g):]

    def generate(self, prompts, dec, max_new, seeds, stop_ids=()):
        self.calls += 1
        mx = max_new if isinstance(max_new, list) else [max_new] * len(prompts)
        assert len(seeds) == len(prompts)
        out = []
        for p, m in zip(prompts, mx):
            head, g = "".join(map(chr, p)).split("[A]", 1)
            ids = [ord(c) for c in (self.rest(g) if head != "N" else "ans\x02")][:m]
            cut = next((k + 1 for k, t in enumerate(ids) if t in {EOS, *stop_ids}), None)
            ids = ids[:cut] if cut else ids
            out.append((ids, [-0.2 if chr(t) in "ans" else -0.05 for t in ids], cut is not None))
        return out


def run(T=40, **kw):
    m = FakeModel(T)
    (ids, lps, forced), = bf.generate(m, [encode("Q[A]")], [("nq0001", "misleading")], {}, 14, True, THINK_END,
                                      encode, **kw)
    reason, ans, alps, trunc = split_answer(ids, lps, THINK_END, {EOS}, True, decode)
    return dict(ids=ids, lps=lps, forced=forced, answer=decode(ans).strip(), conf=confidence(alps), trunc=trunc,
                reasoning=decode(reason), n_think=bf.n_generated(ids, lps, THINK_END), n_gen=bf.n_generated(ids, lps),
                calls=m.calls)


def test_unlimited_is_one_plain_pass():
    r = run(T=40)
    assert r["answer"] == "ans" and not r["forced"] and None not in r["lps"] and r["calls"] == 1
    assert r["n_think"] == 41 and r["n_gen"] == 41 + 1 + 6  # <think> + 40, </think>, "\n\nans" + end of text
    assert math.isclose(r["conf"], math.exp(-0.2))


def test_budget_caps_thinking_and_inserts_qwen_stop_text():
    r = run(T=40, budget=10)
    assert r["n_think"] == 10 and r["forced"] and r["answer"] == "ans" and r["calls"] == 2
    assert "Considering the limited time by the user" in r["reasoning"]
    assert r["n_gen"] == 10 + 4                                   # inserted tokens are not counted as generated
    assert math.isclose(r["conf"], math.exp(-0.2))                # the inserted ".\n\n" is not part of the answer
    assert bf.STOP_TEXT.endswith("now.\n</think>.\n\n")           # verbatim from the Qwen3 report, Sec. 4.3


def test_budget_zero_answers_without_thinking():
    r = run(T=40, budget=0)
    assert r["n_think"] == 0 and r["forced"] and r["answer"] == "ans" and r["n_gen"] == 4 and r["calls"] == 1
    assert r["reasoning"].startswith("<think>\nConsidering")


def test_budget_zero_does_not_open_reasoning_twice():
    m = FakeModel()
    prompt = encode("Q[A]<think>\n")  # a chat template that already opens the reasoning
    (ids, lps, forced), = bf.generate(m, [prompt], [("nq0001", "gold")], {}, 14, True, THINK_END, encode, budget=0)
    assert decode(ids).startswith("Considering") and forced


def test_budget_above_natural_length_changes_nothing():
    a, b = run(T=40), run(T=40, budget=500)
    assert a["ids"] == b["ids"] and a["lps"] == b["lps"] and not b["forced"]


def test_budget_that_cuts_only_the_answer_continues_it():
    r = run(T=40, budget=43)  # reasoning closed at token 42, answer cut: continue without inserting anything
    assert r["answer"] == "ans" and not r["forced"] and None not in r["lps"] and r["n_think"] == 41


def test_wait_extends_thinking_once():
    r = run(T=40, wait=1)
    assert r["n_think"] == 41 + 5 and r["answer"] == "ans" and not r["forced"]
    assert r["lps"].count(None) == len("Wait") and "Wait" in r["reasoning"]


def test_wait_respects_cap():
    r = run(T=40, wait=1, budget=43)  # 41 thinking tokens used, 2 left after "Wait": then the stop text
    assert r["n_think"] <= 43 and r["forced"] and r["answer"] == "ans"


def test_no_thinking_is_a_single_short_pass():
    m = FakeModel()
    (ids, lps, forced), = bf.generate(m, [encode("N[A]")], [("nq0001", "gold")], {}, 14, False, THINK_END, encode)
    assert decode(ids).strip() == "ans" and not forced and m.calls == 1


def test_request_seeds():
    s = bf.request_seed
    assert s(14, "nq0001", "gold") == s(14, "nq0001", "gold")
    assert len({s(14, "nq0001", "gold"), s(15, "nq0001", "gold"), s(14, "nq0002", "gold"),
                s(14, "nq0001", "misleading"), s(14, "nq0001", "gold", 1)}) == 5
    assert all(0 <= s(k, "x", "y") < 2 ** 31 for k in range(50))


def test_run_names_and_configs():
    cfg = dict(model="Qwen/Qwen3-8B", think=True, prompt="base", decoding="recommended", seed=13)
    assert run_name(cfg) == "Qwen3-8B_think"                                   # v1 names unchanged
    assert run_name({**cfg, "prompt": "warn"}) == "Qwen3-8B_think_warn"
    assert run_name({**cfg, "think": False, "seed": 14}) == "Qwen3-8B_nothink_s14"
    assert run_name({**cfg, "budget": 128, "seed": 14}) == "Qwen3-8B_think_b128_s14"
    assert run_name({**cfg, "budget": 0, "seed": 14}) == "Qwen3-8B_think_b0_s14"
    assert run_name({**cfg, "wait": 1, "seed": 14}) == "Qwen3-8B_think_wait1_s14"
    cs = configs("Qwen/Qwen3-8B", [0, 1], ["base"], [None, 128], [14], 0, "recommended", ["gold"], 0)
    assert [run_name(c) for c in cs] == ["Qwen3-8B_nothink_s14", "Qwen3-8B_think_s14", "Qwen3-8B_think_b128_s14"]


def test_split_answer_ignores_inserted_tokens():
    vocab = {1: "</think>", 2: "<eos>", 10: ".", 11: "Sen", 12: "egal"}
    dec = lambda ids: "".join(vocab[i] for i in ids)
    reason, ans, alps, trunc = split_answer([11, 1, 10, 11, 12, 2], [-1, None, None, -0.5, -0.1, 0], 1, {1, 2}, True, dec)
    assert ans == [11, 12] and alps == [-0.5, -0.1] and not trunc


def test_plausible_substitution():
    assert substitute("Dakar is the capital of Senegal. SENEGAL is in Africa.", ["Senegal"], "Mali") == \
        "Dakar is the capital of Mali. Mali is in Africa."
    assert substitute("In 2010 and 10 BC", ["10"], "47") == "In 2010 and 47 BC"            # whole words only
    assert substitute("New South Wales; Wales", ["Wales", "New South Wales"], "X") == "X; X"  # longest first
    assert kind("1957") == "year" and kind("three") == "number" and kind("Bryan Cranston") == "text"
    assert clean_candidate('Wrong answer: "Mali".\nBecause...') == "Mali"
    item = {"org_answer": ["Senegal"], "org_context": "Dakar is the capital of Senegal, in West Africa."}
    assert check(item, "Mali")[0] == "Dakar is the capital of Mali, in West Africa."
    assert check(item, "Republic of Senegal")[1] == "overlaps true answer"
    assert check(item, "Africa")[1] == "already in passage"
    assert check(item, "1957")[1] == "different type"
    assert check(item, "")[1] == "empty"
    it = {"question": "q", "org_answer": ["Senegal"], "sub_answer": ["South Korea"], "plaus_answer": ["Mali"]}
    assert planted_answer(it, "plausible") == ["Mali"] and planted_answer(it, "misleading") == ["South Korea"]
    assert classify("Mali", it["org_answer"], planted_answer(it, "plausible")) == "substituted"


def test_replacement_rule_reproduces_nq_swap():
    path = Path("data/items.jsonl")
    if not path.exists():
        print("  (data/items.jsonl not built: skipped)")
        return
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    same = sum(substitute(it["org_context"], it["org_answer"], it["sub_answer"][0]) == it["sub_context"] for it in items)
    assert same >= 0.95 * len(items), same


def test_data_building_keeps_its_original_normalisation():
    import prepare_data
    from scoring import normalize
    assert prepare_data.normalize("It's 9.21, the Three") == "its 9 21 3"  # as when data/items.jsonl was built
    assert normalize("It's 9.21, the Three") == "its 921 3"                # answer scoring keeps decimals together
    path = Path("data/items.jsonl")
    if path.exists():
        import hashlib
        h = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert h == "0ad33638bebc7e92598fa0ec0829eb0f0527204fb17c096f883e48fb4f76683f", "not the published question set"


def test_agreement_metrics():
    assert cohen_kappa(["y", "n", "y", "n"], ["y", "n", "y", "n"]) == 1.0
    assert math.isclose(cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]), 0.0)
    assert math.isnan(cohen_kappa([], []))
    assert sign_test(0, 0) != sign_test(0, 0)  # nan
    assert math.isclose(sign_test(5, 5), 1.0) and sign_test(10, 0) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
