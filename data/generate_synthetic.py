"""Generate a synthetic RooMatch marketplace with the production schema.

Profiles use the app's real fields and answer options; the swipe deck works the
way the app's does (same city, newest profiles first, pages of 50). What people
*actually* want is a hidden "true preference" model defined in TRUE_EFFECTS below,
which deliberately differs from the app's hand-set weights. The analysis has to
discover those differences from behaviour alone, and the tests check it does.

No names, photos, phone numbers or message text are generated.

    python data/generate_synthetic.py      # writes data/synthetic/*.csv
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomatch.score import components, match_score  # noqa: E402

SEED = 7
START, END = date(2026, 3, 1), date(2026, 10, 1)
N_PROFILES = 6000
DECK_PAGE = 50  # useAllProfiles PAGE_SIZE

# City: (share of users, mean budget ₪, sd)
CITIES = {
    "Tel Aviv": (0.30, 3600, 650), "Jerusalem": (0.17, 2600, 500), "Haifa": (0.12, 2100, 400),
    "Beer Sheva": (0.10, 1700, 300), "Ramat Gan": (0.09, 3200, 550), "Rishon LeZion": (0.06, 2700, 450),
    "Petah Tikva": (0.05, 2600, 450), "Herzliya": (0.04, 3800, 700), "Netanya": (0.05, 2500, 450),
    "Ashdod": (0.02, 2200, 400),
}

# The hidden truth: log-odds effect of each pair feature on a like.
TRUE_EFFECTS = {
    "intercept": -0.55,
    "gender_mismatch": -3.0,     # not in the app's score at all
    "smoking_conflict": -1.4,    # app: 35 pts  (about right)
    "pet_conflict": -0.6,        # app: 20 pts  (about right)
    "religious_gap": -0.30,      # per step     (about right)
    "clean_conflict": -0.30,     # per point    (app underweights)
    "sleep_opposite": -0.9,      # early vs owl (about right)
    "budget_ratio": -1.6,        # continuous   (app underweights)
    "move_in_months": -0.25,     # per 30 days  (app underweights)
    "vibe_same": 0.0,            # app: 10% of base weight (wasted)
    "social_sim": 0.25,          # app: 15% of base weight (overweighted)
    "noise_sim": 0.25,
    "age_fit": 0.5,
}


def sample_profile(rng: np.random.Generator, created: date) -> dict:
    city = rng.choice(list(CITIES), p=[v[0] for v in CITIES.values()])
    _, mu, sd = CITIES[city]
    age = int(np.clip(rng.normal(24.5, 2.6), 19, 33))
    default_age_range = rng.random() < 0.45  # many keep the onboarding default 18-40
    clean = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.30, 0.30, 0.20]))
    gender = str(rng.choice(["Male", "Female"]))
    gp = rng.random()
    gender_pref = ("No preference" if gp < 0.45 else gender if gp < 0.90 else "Mixed")
    return {
        "user_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
        "created_at": created.isoformat(),
        "city": str(city),
        "age": age,
        "gender": gender,
        "gender_preference": gender_pref,
        "preferred_age_min": 18 if default_age_range else max(18, age - int(rng.integers(2, 6))),
        "preferred_age_max": 40 if default_age_range else age + int(rng.integers(2, 7)),
        "budget": int(round(max(900, rng.normal(mu, sd)) / 50) * 50),
        "move_in_date": (created + timedelta(days=int(rng.integers(14, 150)))).isoformat(),
        "sleep_schedule": str(rng.choice(["Early bird", "Night owl", "Flexible"], p=[0.30, 0.35, 0.35])),
        "cleanliness": clean,
        "cleanliness_expectations": int(np.clip(clean + rng.choice([-1, 0, 1, 2], p=[0.15, 0.45, 0.3, 0.1]), 1, 5)),
        "cooking_habits": str(rng.choice(["Daily", "Sometimes", "Rarely", "Never"], p=[0.25, 0.45, 0.22, 0.08])),
        "shared_space_org": str(rng.choice(["Very organized", "Moderate", "Relaxed"], p=[0.3, 0.5, 0.2])),
        "smoking_preference": str(rng.choice(["I smoke", "Non-smoker, okay with smokers",
                                              "Non-smoker, prefer no smoking"], p=[0.15, 0.45, 0.40])),
        "religious_level": str(rng.choice(["Not religious", "Somewhat", "Traditional", "Very religious"],
                                          p=[0.55, 0.20, 0.17, 0.08])),
        "alcohol": str(rng.choice(["Never", "Socially", "Often"], p=[0.15, 0.70, 0.15])),
        "pet_preference": str(rng.choice(["I have pets", "No pets, but okay with them", "Prefer no pets"],
                                         p=[0.12, 0.55, 0.33])),
        "guests": str(rng.choice(["Frequently", "Occasionally", "Rarely", "Never"], p=[0.15, 0.45, 0.32, 0.08])),
        "partying": str(rng.choice(["Frequently", "Occasionally", "Rarely", "Never"], p=[0.12, 0.38, 0.35, 0.15])),
        "lifestyle_vibe": str(rng.choice(["Chill & quiet", "Balanced", "Social & active", "Party lover"],
                                         p=[0.30, 0.42, 0.22, 0.06])),
        "activity_hours": str(rng.choice(["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)",
                                          "Night (24-6)", "Flexible"], p=[0.15, 0.10, 0.25, 0.10, 0.40])),
        "social_level": int(rng.integers(1, 6)),
        "noise_level": int(rng.integers(1, 6)),
        "wfh": bool(rng.random() < 0.3),
    }


def pair_features(a: dict, b: dict) -> dict:
    """Features of a (viewer a, target b) pair, including ones the app ignores."""
    c = components(a, b)

    def wants(viewer, target):
        p = viewer["gender_preference"]
        return p in ("No preference", "Mixed") or p == target["gender"]

    return {
        "match_percent": match_score(a, b),
        "gender_mismatch": not wants(a, b),
        "smoking_conflict": bool(c["smoking_conflict"]),
        "pet_conflict": bool(c["pet_conflict"]),
        "religious_gap": c["religious_gap"],
        "clean_conflict": c["clean_conflict"],
        "sleep_opposite": c["sleep_gap"] == 2,
        "budget_ratio": c["budget_ratio"],
        "move_in_months": (c["move_in_days"] or 0) / 30,
        "vibe_same": c["similarity"]["vibe"] == 1.0,
        "social_sim": c["similarity"]["social"],
        "noise_sim": c["similarity"]["noise"],
        "age_fit": c["similarity"].get("age_fit", 0.5),
    }


def true_logit(f: dict, attract: float, picky: float) -> float:
    return attract + picky + sum(TRUE_EFFECTS[k] * float(f[k]) if k != "intercept" else TRUE_EFFECTS[k]
                                 for k in TRUE_EFFECTS)


def simulate():
    rng = np.random.default_rng(SEED)
    span = (END - START).days
    created = sorted(START + timedelta(days=int(span * rng.beta(1.5, 1.0))) for _ in range(N_PROFILES))
    profiles = [sample_profile(rng, d) for d in created]
    attract = rng.normal(0, 0.7, N_PROFILES)     # photos, bio... unobserved by the analysis
    picky = rng.normal(0, 0.5, N_PROFILES)
    by_id = {p["user_id"]: i for i, p in enumerate(profiles)}

    # Sessions: on join day, then a decaying number of weekly returns.
    sessions = []
    for i, p in enumerate(profiles):
        d0 = date.fromisoformat(p["created_at"])
        n_return = rng.geometric(0.30) - 1
        days = [d0] + sorted(d0 + timedelta(days=int(rng.integers(3, 60))) for _ in range(n_return))
        sessions += [(d, i) for d in days if d < END]
    sessions.sort()

    swiped: dict[int, set[int]] = {i: set() for i in range(N_PROFILES)}
    likes = set()
    swipe_rows, like_rows = [], []
    city_members: dict[str, list[int]] = {c: [] for c in CITIES}
    joined = 0
    for d, i in sessions:
        while joined < N_PROFILES and date.fromisoformat(profiles[joined]["created_at"]) <= d:
            city_members[profiles[joined]["city"]].append(joined)
            joined += 1
        me = profiles[i]
        # Deck: same city, newest first, one page, minus anyone already swiped.
        deck = [j for j in reversed(city_members[me["city"]]) if j != i and j not in swiped[i]][:DECK_PAGE]
        n_swipes = min(len(deck), int(rng.geometric(1 / 22)))
        t = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=float(rng.uniform(8, 23)))
        for pos, j in enumerate(deck[:n_swipes]):
            f = pair_features(me, profiles[j])
            liked = rng.random() < 1 / (1 + np.exp(-true_logit(f, attract[j], picky[i])))
            swiped[i].add(j)
            at = t + timedelta(seconds=6 * pos)
            swipe_rows.append({"viewer_id": me["user_id"], "target_id": profiles[j]["user_id"],
                               "direction": "like" if liked else "pass", "match_percent": f["match_percent"],
                               "deck_position": pos, "swiped_at": at})
            if liked:
                likes.add((i, j))
                like_rows.append({"user_id": me["user_id"], "liked_user_id": profiles[j]["user_id"],
                                  "created_at": at})

    # Mutual likes -> conversations -> messages (metadata only, never content).
    conv_rows, part_rows, msg_rows = [], [], []
    like_time = {(r["user_id"], r["liked_user_id"]): r["created_at"] for r in like_rows}
    for (i, j) in likes:
        if i < j and (j, i) in likes:
            a, b = profiles[i], profiles[j]
            matched_at = max(like_time[(a["user_id"], b["user_id"])], like_time[(b["user_id"], a["user_id"])])
            if rng.random() > 0.72:
                continue
            cid = str(uuid.UUID(bytes=rng.bytes(16), version=4))
            conv_rows.append({"id": cid, "created_at": matched_at + timedelta(minutes=float(rng.exponential(300)))})
            part_rows += [{"conversation_id": cid, "user_id": a["user_id"]},
                          {"conversation_id": cid, "user_id": b["user_id"]}]
            # Whether it turns into a real conversation depends on the same true fit.
            fa, fb = pair_features(a, b), pair_features(b, a)
            fit = (true_logit(fa, 0, 0) + true_logit(fb, 0, 0)) / 2
            n_msgs = int(rng.poisson(np.exp(1.2 + 0.45 * fit))) + 1
            t = conv_rows[-1]["created_at"]
            sender = rng.integers(0, 2)
            for _ in range(n_msgs):
                t += timedelta(minutes=float(rng.exponential(90)))
                msg_rows.append({"conversation_id": cid, "sender_id": (a, b)[sender]["user_id"], "created_at": t})
                sender = 1 - sender if rng.random() < 0.8 else sender

    prof = pd.DataFrame(profiles)
    return (prof, pd.DataFrame(swipe_rows), pd.DataFrame(like_rows), pd.DataFrame(conv_rows),
            pd.DataFrame(part_rows), pd.DataFrame(msg_rows), by_id)


if __name__ == "__main__":
    out = Path(__file__).parent / "synthetic"
    out.mkdir(exist_ok=True)
    prof, swipes, likes, convs, parts, msgs, _ = simulate()
    for name, df in [("profiles", prof), ("swipes", swipes), ("liked_profiles", likes),
                     ("conversations", convs), ("conversation_participants", parts), ("messages", msgs)]:
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"{len(prof):,} profiles, {len(swipes):,} swipes, {len(likes):,} likes, "
          f"{len(convs):,} conversations, {len(msgs):,} messages -> {out}")
