"""RooMatch matching analysis, end to end.

    python analysis/run_analysis.py

1. SQL (sql/*.sql, DuckDB) turns the event tables into aggregate tables.
2. A pair-level model (Python) asks which compatibility factors actually drive likes,
   using the exact production score ported in roomatch/score.py.
3. Everything is written to analysis/results.json, analysis/findings.md and figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from statsmodels.stats.power import NormalIndPower  # noqa: E402
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from generate_synthetic import pair_features  # noqa: E402

DATA, SQL, FIG, OUT = ROOT / "data" / "synthetic", ROOT / "sql", ROOT / "figures", ROOT / "analysis"
TABLES = ["profiles", "swipes", "liked_profiles", "conversations", "conversation_participants", "messages"]
FEATURES = ["gender_mismatch", "smoking_conflict", "pet_conflict", "religious_gap", "clean_conflict",
            "sleep_opposite", "budget_ratio", "move_in_months", "vibe_same", "social_sim", "noise_sim", "age_fit"]
APP_FEATURES = [f for f in FEATURES if f != "gender_mismatch"]

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOTE = "Synthetic data with the production schema and the production scoring code; not real users."
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name):
    fig.text(0.01, 0.01, NOTE, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def wilson(k, n):
    lo, hi = proportion_confint(np.asarray(k), np.asarray(n), method="wilson")
    return np.asarray(lo), np.asarray(hi)


# --------------------------------------------------------------------- Q1

def q1_score_vs_outcomes(bands: pd.DataFrame, swipes: pd.DataFrame) -> dict:
    b = bands.copy()
    b["like_rate"] = b.likes / b.swipes
    b["mutual_rate"] = b.mutual_pairs / b.pairs_both_saw
    b["chat_rate"] = b.two_way_chats / b.pairs_both_saw
    out = {"bands": b.to_dict(orient="records"),
           "auc_like": float(roc_auc_score(swipes.direction == "like", swipes.match_percent)),
           "floor_share_of_cards": float(b.loc[b.band == "15 (floor)", "swipes"].sum() / b.swipes.sum())}

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(len(b))
    lo, hi = wilson(b.mutual_pairs, b.pairs_both_saw)
    colors = [GREY if band.startswith("15") else BLUE for band in b.band]
    ax.bar(x, b.mutual_rate * 100, color=colors, width=0.6,
           yerr=[(b.mutual_rate - lo) * 100, (hi - b.mutual_rate) * 100],
           error_kw={"ecolor": INK2, "capsize": 4, "lw": 1})
    for i, v in enumerate(b.mutual_rate * 100):
        ax.text(i, hi[i] * 100 + 0.4, f"{v:.1f}%", ha="center", color=INK)
    ax.set_xticks(x, [f"{band}%" if not band.startswith("15") else "15%\n(floor)" for band in b.band])
    ax.set_xlabel('"Match %" shown to users')
    ax.set_ylabel("Mutual-like rate (%, 95% CI)")
    ax.set_ylim(0, max(hi) * 100 * 1.2)
    ax.set_title('The score ranks well, but "70% match" means about an 8% chance of matching')
    ax.grid(axis="x", visible=False)
    save(fig, "01_mutual_rate_by_score.png")
    return out


# --------------------------------------------------------------------- Q2

def build_pairs(profiles: pd.DataFrame, swipes: pd.DataFrame) -> pd.DataFrame:
    prof = profiles.set_index("user_id").to_dict(orient="index")
    for u, p in prof.items():
        p["user_id"] = u
    rows = [pair_features(prof[v], prof[t]) for v, t in zip(swipes.viewer_id, swipes.target_id)]
    X = pd.DataFrame(rows).astype(float)
    X["liked"] = (swipes.direction == "like").to_numpy().astype(int)
    X["viewer_id"] = swipes.viewer_id.to_numpy()
    X["target_id"] = swipes.target_id.to_numpy()
    # Popularity control: the target's like rate from *other* viewers (leave-one-out),
    # so a model can't credit compatibility for what is really an attractive profile.
    g = X.groupby("target_id").liked
    n, k = g.transform("count"), g.transform("sum")
    prior = X.liked.mean()
    X["target_popularity"] = ((k - X.liked) + prior * 10) / ((n - 1) + 10)
    X["target_popularity"] = np.log(X.target_popularity / (1 - X.target_popularity))
    return X


def fit(X: pd.DataFrame, cols: list[str]):
    return sm.Logit(X.liked, sm.add_constant(X[cols])).fit(disp=0)


# A "typical conflict" per factor, so the app's points and the data's effect are
# compared on the same situation. (feature changes from -> to, app point cost)
STANDARD_CONFLICTS = {
    "Gender preference not met": ("gender_mismatch", 0, 1, 0),
    "Smoker vs. minds smoking": ("smoking_conflict", 0, 1, 35),
    "Budgets 50% apart": ("budget_ratio", 0.0, 0.5, 7),
    "Cleanliness gap of 3": ("clean_conflict", 0, 3, 14),
    "Early bird vs. night owl": ("sleep_opposite", 0, 1, 18),
    "Move-in 4 months apart": ("move_in_months", 0.5, 4.0, 16),
    "Religious gap of 2": ("religious_gap", 0, 2, 18),
    "Pets vs. minds pets": ("pet_conflict", 0, 1, 20),
    "Different lifestyle vibe": ("vibe_same", 1, 0, 12),
    "Social level 2 apart": ("social_sim", 1.0, 0.5, 9),
}


def q2_what_drives_likes(X: pd.DataFrame) -> dict:
    rng = np.random.default_rng(0)
    viewers = X.viewer_id.unique()
    test_viewers = set(rng.choice(viewers, size=int(0.3 * len(viewers)), replace=False))
    is_test = X.viewer_id.isin(test_viewers).to_numpy()
    train, test = X[~is_test], X[is_test]

    full = fit(train, FEATURES + ["target_popularity"])
    app_reweighted = fit(train, APP_FEATURES + ["target_popularity"])
    app_only = fit(train, ["match_percent", "target_popularity"])

    def auc(model, cols):
        return float(roc_auc_score(test.liked, model.predict(sm.add_constant(test[cols]))))

    out = {
        "n_swipes": int(len(X)), "n_test_swipes": int(is_test.sum()),
        "auc_score_alone": float(roc_auc_score(test.liked, test.match_percent)),
        "auc_score_plus_popularity": auc(app_only, ["match_percent", "target_popularity"]),
        "auc_reweighted_app_factors": auc(app_reweighted, APP_FEATURES + ["target_popularity"]),
        "auc_reweighted_plus_gender": auc(full, FEATURES + ["target_popularity"]),
        "coefficients": {k: float(v) for k, v in full.params.items()},
        "coef_ci95": {k: [float(a), float(b)] for k, (a, b) in full.conf_int().iterrows()},
        "gender_mismatch_share_of_cards": float(X.gender_mismatch.mean()),
    }

    # Average effect of each standard conflict on like probability (percentage points).
    base = X[FEATURES + ["target_popularity"]].sample(20000, random_state=1)
    effects = {}
    for label, (col, frm, to, app_pts) in STANDARD_CONFLICTS.items():
        a, b = base.copy(), base.copy()
        a[col], b[col] = frm, to
        pa = full.predict(sm.add_constant(a, has_constant="add"))
        pb = full.predict(sm.add_constant(b, has_constant="add"))
        effects[label] = {"data_pp": float((pb - pa).mean() * 100), "app_points": app_pts}
    out["standard_conflicts"] = effects

    # Chart: importance on a common scale, anchored on smoking (= 100 on both sides),
    # a deal-breaker the app and users both treat as serious.
    labels = list(STANDARD_CONFLICTS)
    anchor = "Smoker vs. minds smoking"
    data_imp = np.array([max(0.0, -effects[k]["data_pp"]) for k in labels]) / -effects[anchor]["data_pp"] * 100
    app_imp = np.array([effects[k]["app_points"] for k in labels], dtype=float) / effects[anchor]["app_points"] * 100
    for k, d, a in zip(labels, data_imp, app_imp):
        effects[k]["data_index"], effects[k]["app_index"] = float(d), float(a)
        effects[k]["verdict"] = ("missing" if a == 0 else "under-weighted" if d > 1.3 * a
                                 else "over-weighted" if d < 0.7 * a else "about right")
    order = np.argsort(data_imp)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(labels))
    for yi, i in enumerate(order):
        ax.plot([app_imp[i], data_imp[i]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(app_imp[order], y, color=GREY, s=60, zorder=2, label="What the app assumes (penalty points)")
    ax.scatter(data_imp[order], y, color=BLUE, s=60, zorder=3, label="What users actually do (effect on likes)")
    ax.set_yticks(y, [labels[i] for i in order])
    ax.axvline(100, color=GRID, lw=1, ls="--", zorder=0)
    ax.set_xlabel("Importance, indexed to smoking conflict = 100")
    ax.set_xlim(-5, max(data_imp.max(), app_imp.max()) * 1.08)
    ax.set_title("The biggest deal-breaker isn't in the score at all")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "02_app_weights_vs_behaviour.png")
    return out, full


# --------------------------------------------------------------------- Q3

def q3_liquidity(unseen: pd.DataFrame, exposure: pd.DataFrame, X: pd.DataFrame, model,
                 profiles: pd.DataFrame, swipes: pd.DataFrame, mutual_pairs: int) -> dict:
    u = unseen.iloc[0]
    out = {"likes": int(u.likes), "unseen_share": float(u.target_never_saw_liker / u.likes),
           "liked_back_share_when_seen": float(u.liked_back / (u.liked_back + u.passed_back))}
    e = exposure.copy()
    e["per_profile"] = e.impressions / e.eligible_profiles
    out["first_week_share_of_impressions"] = float(e.impressions.iloc[0] / e.impressions.sum())
    out["impressions_per_profile_by_week"] = dict(zip(e.weeks_since_join.astype(int), e.per_profile.round(2)))

    # Model-based estimate: if each unseen like were shown back to its target first
    # (a "they liked you" queue), how many extra mutual likes would we expect?
    prof = profiles.set_index("user_id").to_dict(orient="index")
    for k, p in prof.items():
        p["user_id"] = k
    seen = set(zip(swipes.viewer_id, swipes.target_id))
    likes = swipes[swipes.direction == "like"]
    # Only targets who came back after the like could have been shown a queue.
    last_active = pd.to_datetime(swipes.swiped_at).groupby(swipes.viewer_id).max()
    liked_at = pd.to_datetime(likes.swiped_at)
    pend = [(v, t) for v, t, at in zip(likes.viewer_id, likes.target_id, liked_at)
            if (t, v) not in seen and t in last_active.index and last_active[t] > at]
    out["unseen_likes_target_returned"] = len(pend)
    rows = pd.DataFrame([pair_features(prof[t], prof[v]) for v, t in pend]).astype(float)  # target looks at liker
    pop = X.groupby("target_id").target_popularity.first()
    rows["target_popularity"] = [pop.get(v, pop.median()) for v, _ in pend]
    expected = float(model.predict(sm.add_constant(rows[FEATURES + ["target_popularity"]], has_constant="add")).sum())
    out["likes_you_queue_expected_new_mutuals"] = expected
    out["current_mutual_pairs"] = int(mutual_pairs)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar([f"wk {w}" if w else "join\nweek" for w in e.weeks_since_join.astype(int)], e.per_profile,
           color=[BLUE] + [GREY] * (len(e) - 1), width=0.6)
    for i, v in enumerate(e.per_profile):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", color=INK)
    ax.set_ylabel("Times shown per profile")
    ax.set_title(f"Newest-first ordering: {out['first_week_share_of_impressions'] * 100:.0f}% of a profile's "
                 "exposure is in its first week")
    ax.grid(axis="x", visible=False)
    save(fig, "03_exposure_decay.png")
    return out


# --------------------------------------------------------------------- Q4

def q4_cities(c: pd.DataFrame) -> dict:
    c = c.copy()
    c["rate"] = c.matched_within_30d / c.new_users
    lo, hi = wilson(c.matched_within_30d, c.new_users)
    c["lo"], c["hi"] = lo, hi
    c = c.sort_values("rate")
    corr = float(np.corrcoef(c.city_pool, c.rate)[0, 1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(c))
    ax.barh(y, c.rate * 100, color=[ORANGE if city == "Tel Aviv" else BLUE for city in c.city], height=0.6,
            xerr=[(c.rate - c.lo) * 100, (c.hi - c.rate) * 100], error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    ax.set_yticks(y, [f"{city}  (pool {pool:,})" for city, pool in zip(c.city, c.city_pool)])
    ax.set_xlabel("New users with a mutual like within 30 days (%, 95% CI)")
    ax.set_title("A bigger city doesn't mean more matches")
    ax.grid(axis="y", visible=False)
    save(fig, "04_city_liquidity.png")
    return {"by_city": c[["city", "new_users", "matched_within_30d", "rate", "lo", "hi", "city_pool",
                          "median_budget"]].to_dict(orient="records"),
            "pool_vs_rate_correlation": corr}


def write_findings(r: dict) -> None:
    q1, q2, q3, q4 = r["q1"], r["q2"], r["q3"], r["q4"]
    band = {b["band"]: b for b in q1["bands"]}
    sc = q2["standard_conflicts"]
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    tlv = next(x for x in q4["by_city"] if x["city"] == "Tel Aviv")
    best = max(q4["by_city"], key=lambda x: x["rate"])
    lines = [
        "# Findings (synthetic data)", "",
        "_Generated by `analysis/run_analysis.py` from `results.json`._", "",
        "## Q1: Does the score predict who connects?",
        f"- Mutual-like rate rises steadily with the score: {pct(band['15 (floor)']['mutual_rate'])} at the "
        f"15% floor → {pct(band['75+']['mutual_rate'])} at 75%+. AUC for a like: {q1['auc_like']:.3f}.",
        f"- But it is not a probability: pairs shown as \"60-74% match\" became mutual {pct(band['60-74']['mutual_rate'])} "
        "of the time.",
        f"- {pct(q1['floor_share_of_cards'])} of all cards shown sat at the 15% floor. The code comment says these are "
        "suppressed, but nothing filters them.", "",
        "## Q2: Which factors actually drive likes?",
        f"- Model on {q2['n_swipes']:,} swipes; tested on users the model never saw ({q2['n_test_swipes']:,} swipes).",
        f"- Gender preference not met: **{sc['Gender preference not met']['data_pp']:+.1f} pp** like probability. "
        f"It is the largest effect, yet weighted zero in the score and not filtered. "
        f"{pct(q2['gender_mismatch_share_of_cards'])} of cards shown break the viewer's stated preference.",
        *[f"- {verdict.capitalize()} by the app: " + "; ".join(
            f"{k.lower()} {v['data_pp']:+.1f} pp (index {v['data_index']:.0f} vs app {v['app_index']:.0f})"
            for k, v in sc.items() if v["verdict"] == verdict)
          for verdict in ("under-weighted", "about right", "over-weighted")
          if any(v["verdict"] == verdict for v in sc.values())],
        f"- AUC on held-out users: score alone {q2['auc_score_alone']:.3f} → re-weighted app factors "
        f"{q2['auc_reweighted_app_factors']:.3f} → plus gender preference {q2['auc_reweighted_plus_gender']:.3f}.", "",
        "## Q3: Are good matches getting lost?",
        f"- {pct(q3['unseen_share'])} of likes go to someone who never sees the liker, so they can never become a match.",
        f"- When the target does see the liker, they like back {pct(q3['liked_back_share_when_seen'])} of the time.",
        f"- {pct(q3['first_week_share_of_impressions'])} of a profile's impressions happen in its first week.",
        f"- {q3['unseen_likes_target_returned']:,} of those unseen likes went to someone who opened the app again later. "
        f"Showing them a \"they liked you\" queue is expected to add about "
        f"{q3['likes_you_queue_expected_new_mutuals']:,.0f} mutual likes on top of {q3['current_mutual_pairs']:,} today "
        "(model estimate, to be tested).", "",
        "## Q4: Is the market liquid everywhere?",
        f"- Tel Aviv, the largest pool, gets {pct(tlv['rate'])} of new users a mutual like within 30 days, against "
        f"{pct(best['rate'])} in {best['city']}. Correlation between pool size and match rate: "
        f"{q4['pool_vs_rate_correlation']:+.2f}.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    if not (DATA / "swipes.csv").exists():
        raise SystemExit("Run `python data/generate_synthetic.py` first.")
    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"create view {t} as select * from read_csv_auto('{DATA / f'{t}.csv'}')")
    q = {f.stem: con.execute(f.read_text()).df() for f in sorted(SQL.glob("*.sql"))}
    agg = OUT / "aggregates"
    agg.mkdir(exist_ok=True)
    for name, df in q.items():
        df.to_csv(agg / f"{name}.csv", index=False)

    profiles = pd.read_csv(DATA / "profiles.csv")
    swipes = pd.read_csv(DATA / "swipes.csv")
    X = build_pairs(profiles, swipes)

    res = {"q1": q1_score_vs_outcomes(q["01_outcomes_by_score_band"], swipes)}
    res["q2"], model = q2_what_drives_likes(X)
    res["q3"] = q3_liquidity(q["02_unseen_likes"], q["03_exposure_by_profile_age"], X, model, profiles, swipes,
                             int(q["01_outcomes_by_score_band"].mutual_pairs.sum()))
    res["q4"] = q4_cities(q["04_city_liquidity"])
    baseline = float(np.average([c["rate"] for c in res["q4"]["by_city"]],
                                weights=[c["new_users"] for c in res["q4"]["by_city"]]))
    res["experiment"] = {
        "baseline_matched_30d": baseline,
        "n_per_arm_for_5pp": float(NormalIndPower().solve_power(
            proportion_effectsize(baseline + 0.05, baseline), alpha=0.05, power=0.8)),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    write_findings(res)
    print((OUT / "findings.md").read_text())


if __name__ == "__main__":
    main()
