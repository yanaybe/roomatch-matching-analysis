# Privacy

RooMatch's production data is personal and, in places, sensitive (religious level,
smoking, gender, private messages). The rules for this repository:

- **Synthetic only.** Nothing here comes from a real user. The generator produces no
  names, photos, phone numbers, bios or message text, not even fake ones.
- **Message content is never needed.** Matching quality is measured from message
  *metadata* (who sent, when). The synthetic `messages` table has no content column,
  and a test enforces it.
- **Only aggregates are published.** Every table in `analysis/aggregates/` is a count by
  band, city or week, and a test fails if an id column appears in one.
- **Running on production.** The pair-level model needs row-level swipes, so it runs only
  inside the production environment. Only coefficients and aggregates (cells under 10
  suppressed) leave it.
- **Sensitive attributes are analysed as stated preferences, not identities.** The
  question is always "does a mismatch in what two people *said they want* predict
  whether they connect", never a judgement about any group.
