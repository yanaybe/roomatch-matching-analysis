"""Python port of RooMatch's production compatibility score (`calculateUserMatch`).

The app shows this number to users as "NN% match". Porting it lets the analysis
(1) recompute the exact score for any pair and (2) break it into the components
the app's hand-set weights combine. `tests/test_score_port.py` checks the port
against the production TypeScript on thousands of random pairs.
"""

from __future__ import annotations

import math
from datetime import date

FLOOR = 15  # MATCH_SUPPRESSION_THRESHOLD: scores are clamped here, not hidden

FREQUENCY = ["Never", "Rarely", "Sometimes", "Occasionally", "Often", "Frequently", "Daily"]
RELIGIOUS = ["Not religious", "Somewhat", "Traditional", "Very religious"]
ALCOHOL = ["Never", "Socially", "Often"]
ORG = ["Relaxed", "Moderate", "Very organized"]
SLEEP = ["Early bird", "Flexible", "Night owl"]


def js_round(x: float) -> int:
    """JavaScript Math.round: halves round towards +infinity (Python rounds to even)."""
    return math.floor(x + 0.5)


def _idx(scale, v):
    return scale.index(v) if v in scale else -1


def _numeric(a, b, mx):
    return 0.5 if a is None or b is None else 1 - abs(a - b) / mx


def _ordinal(a, b, scale):
    ai, bi = _idx(scale, a), _idx(scale, b)
    return 0.5 if ai == -1 or bi == -1 else 1 - abs(ai - bi) / (len(scale) - 1)


def _days_apart(a: str, b: str) -> float:
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def components(a: dict, b: dict) -> dict:
    """Every input the production score uses, as named features (penalties in points)."""
    smokes = lambda p: p["smoking_preference"] == "I smoke"  # noqa: E731
    minds_smoke = lambda p: p["smoking_preference"] == "Non-smoker, prefer no smoking"  # noqa: E731
    has_pets = lambda p: p["pet_preference"] == "I have pets"  # noqa: E731
    minds_pets = lambda p: p["pet_preference"] == "Prefer no pets"  # noqa: E731

    c: dict = {}
    c["smoking_conflict"] = (smokes(a) and minds_smoke(b)) or (smokes(b) and minds_smoke(a))
    c["pet_conflict"] = (has_pets(a) and minds_pets(b)) or (has_pets(b) and minds_pets(a))

    ra, rb = _idx(RELIGIOUS, a["religious_level"]), _idx(RELIGIOUS, b["religious_level"])
    c["religious_gap"] = abs(ra - rb) if ra != -1 and rb != -1 else 0
    rel_pen = 0
    if ra != -1 and rb != -1 and c["religious_gap"] > 0:
        gap = c["religious_gap"]
        base = 8 if gap == 1 else 20 if gap == 2 else 35
        amp = 0.5 + 0.5 * (max(ra, rb) / (len(RELIGIOUS) - 1))
        rel_pen = js_round(base * amp)

    c["clean_conflict"] = (max(0, a["cleanliness_expectations"] - b["cleanliness"])
                           + max(0, b["cleanliness_expectations"] - a["cleanliness"]))
    cc = c["clean_conflict"]
    clean_pen = 22 if cc >= 5 else 14 if cc >= 3 else 8 if cc >= 2 else 3 if cc >= 1 else 0

    sa, sb = _idx(SLEEP, a["sleep_schedule"]), _idx(SLEEP, b["sleep_schedule"])
    c["sleep_gap"] = abs(sa - sb) if sa != -1 and sb != -1 else 0
    sleep_pen = 18 if c["sleep_gap"] == 2 else 5 if c["sleep_gap"] == 1 else 0

    ba, bb = a.get("budget") or 0, b.get("budget") or 0
    c["budget_ratio"] = abs(ba - bb) / ((ba + bb) / 2) if ba > 0 and bb > 0 else 0.0
    br = c["budget_ratio"]
    budget_pen = 12 if br > 0.6 else 7 if br > 0.4 else 3 if br > 0.25 else 0

    move_pen = 0
    c["move_in_days"] = None
    if a.get("move_in_date") and b.get("move_in_date"):
        d = _days_apart(a["move_in_date"], b["move_in_date"])
        c["move_in_days"] = d
        move_pen = -8 if d <= 30 else -4 if d <= 60 else 8 if d > 90 else 0

    c["penalty_points"] = {
        "smoking": 35 * c["smoking_conflict"], "pets": 20 * c["pet_conflict"], "religion": rel_pen,
        "cleanliness": clean_pen, "sleep": sleep_pen, "budget": budget_pen, "move_in": move_pen,
    }

    a_flex, b_flex = a["activity_hours"] == "Flexible", b["activity_hours"] == "Flexible"
    act = (1 if a_flex and b_flex else 0.8 if a_flex or b_flex
           else 1 if a["activity_hours"] == b["activity_hours"] else 0.3)
    sims = {
        "social": (15, _numeric(a["social_level"], b["social_level"], 4)),
        "noise": (13, _numeric(a["noise_level"], b["noise_level"], 4)),
        "vibe": (10, 1.0 if a["lifestyle_vibe"] == b["lifestyle_vibe"] else 0.0),
        "guests": (8, _ordinal(a["guests"], b["guests"], FREQUENCY)),
        "partying": (8, _ordinal(a["partying"], b["partying"], FREQUENCY)),
        "wfh": (7, 1.0 if a["wfh"] == b["wfh"] else 0.6),
        "cooking": (6, _ordinal(a["cooking_habits"], b["cooking_habits"], FREQUENCY)),
        "activity": (5, act),
        "shared_space": (4, _ordinal(a["shared_space_org"], b["shared_space_org"], ORG)),
        "alcohol": (4, _ordinal(a["alcohol"], b["alcohol"], ALCOHOL)),
    }
    if a.get("age") and b.get("age"):
        a_in = a["preferred_age_min"] <= b["age"] <= a["preferred_age_max"]
        b_in = b["preferred_age_min"] <= a["age"] <= b["preferred_age_max"]
        sims["age_fit"] = (3, 0.5 * a_in + 0.5 * b_in)
    c["similarity"] = {k: s for k, (_, s) in sims.items()}
    total_w = sum(w for w, _ in sims.values())
    c["base_score"] = sum(w * s for w, s in sims.values()) / total_w * 100
    return c


def match_score(a: dict, b: dict) -> int:
    """Exactly what the app displays: max(15, round(base - penalties))."""
    c = components(a, b)
    return max(FLOOR, js_round(c["base_score"] - sum(c["penalty_points"].values())))
