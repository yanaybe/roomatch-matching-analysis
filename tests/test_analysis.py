"""Does the analysis recover the truth we planted, and nothing else?

`data/generate_synthetic.py` decides how users really choose (TRUE_EFFECTS). The
analysis never reads those numbers; it only sees swipes. If it can't recover them
here, where the answer is known, it shouldn't be trusted on real data.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from generate_synthetic import TRUE_EFFECTS  # noqa: E402


@pytest.fixture(scope="module")
def results():
    if not (ROOT / "data" / "synthetic" / "swipes.csv").exists():
        pytest.skip("run `python data/generate_synthetic.py` first")
    subprocess.run([sys.executable, str(ROOT / "analysis" / "run_analysis.py")], check=True, capture_output=True)
    return json.loads((ROOT / "analysis" / "results.json").read_text())


@pytest.mark.parametrize("factor", [k for k, v in TRUE_EFFECTS.items() if k != "intercept" and v != 0])
def test_recovers_each_planted_effect(results, factor):
    est, truth = results["q2"]["coefficients"][factor], TRUE_EFFECTS[factor]
    assert est * truth > 0, "wrong sign"
    # Unobserved viewer pickiness shrinks logistic coefficients towards zero, so allow 30%.
    assert abs(est - truth) <= 0.30 * abs(truth)


def test_finds_no_effect_where_none_was_planted(results):
    lo, hi = results["q2"]["coef_ci95"]["vibe_same"]
    assert lo < 0 < hi


def test_reweighting_beats_the_production_score_on_unseen_users(results):
    q = results["q2"]
    assert q["auc_reweighted_app_factors"] > q["auc_score_alone"]
    assert q["auc_reweighted_plus_gender"] > q["auc_reweighted_app_factors"]


def test_published_tables_are_aggregates_only(results):
    for csv in (ROOT / "analysis" / "aggregates").glob("*.csv"):
        cols = pd.read_csv(csv).columns
        assert not any(c == "id" or c.endswith("_id") for c in cols), csv.name
