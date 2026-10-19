# Dashboards: what RooMatch tracks today, and what it should track

## Today: the admin dashboard

RooMatch has an admin area I built (`/admin`) with seven sections.

| Who | What they use it for | Sections |
|---|---|---|
| **Founder / product** | "Are we growing?" Sign-ups, onboarding completion, listings, premium conversion | Overview, Analytics |
| **Trust & safety** | Handling reports, banning abusive users, reviewing listings | Reports, Users, Rooms |
| **Developer** | Webhook and system health, audit logs | Logs, System |

**What's missing:** every metric there measures **supply** (how many users, how many
listings) and none measures **whether the product works**, that is, whether people find
a roommate. A city can grow 50% in sign-ups while its users match less (see Q4).

## Proposed: a "match health" view

Built from this analysis. Each metric names its denominator, because the wrong
denominator was the main trap here.

| Metric | Definition | Why | Alert when |
|---|---|---|---|
| **Matched within 30 days** (north star) | New users with ≥1 mutual like within 30 days ÷ new users who joined 30+ days ago | Did the product do its job for a new user? | Drops 3 pp week over week |
| **Like-back visibility** | Likes whose target later saw the liker ÷ all likes | Measures the "lost likes" problem directly (78% unseen today) | Below 50% once the queue ships |
| **Mutual rate by score tier** | Mutual likes ÷ pairs who both saw each other, per tier | Is the score still ranking well? Catches drift after re-weighting | Top tier ÷ bottom tier < 5× |
| **Wasted cards** | Cards breaking a stated preference (gender, smoking) ÷ cards shown | Measures deck quality directly | Above 5% |
| **Exposure fairness** | Impressions per profile in weeks 2–4 ÷ week 1 | Detects older profiles being buried | Below 0.2 |
| **Two-way chat rate** | Conversations where both sent ≥ 2 messages ÷ mutual likes | Do matches turn into real contact? | Drops 5 pp |
| All of the above **by city** | — | Liquidity differs by city; averages hide it | — |

All of these become computable once passes are logged with `target_id` and
`match_percent` (recommendation #5). The queries in [`sql/`](sql) are the
starting point for each.
