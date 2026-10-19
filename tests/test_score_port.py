"""The Python score must equal the production TypeScript score, pair for pair.

The fixture holds 1,500 random profile pairs (including missing budgets, ages and
move-in dates) and the scores the app's own `calculateUserMatch` gave them.
"""
import json
from pathlib import Path

from roomatch.score import js_round, match_score

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ts_scores.json").read_text())


def test_port_matches_production_typescript_exactly():
    got = [match_score(a, b) for a, b in FIXTURE["pairs"]]
    mismatches = [i for i, (g, t) in enumerate(zip(got, FIXTURE["scores"])) if g != t]
    assert not mismatches, f"{len(mismatches)} of {len(got)} pairs differ, e.g. #{mismatches[:3]}"


def test_js_rounding_semantics():
    assert [js_round(x) for x in (0.5, 1.5, 2.5, -0.5, -1.5)] == [1, 2, 3, 0, -1]
