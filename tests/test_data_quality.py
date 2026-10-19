"""Checks I'd run on the production tables before trusting any matching metric."""
from pathlib import Path

import duckdb
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
TABLES = ["profiles", "swipes", "liked_profiles", "conversations", "conversation_participants", "messages"]


@pytest.fixture(scope="module")
def con():
    if not (DATA / "swipes.csv").exists():
        pytest.skip("run `python data/generate_synthetic.py` first")
    c = duckdb.connect()
    for t in TABLES:
        c.execute(f"create view {t} as select * from read_csv_auto('{DATA / f'{t}.csv'}')")
    return c


def one(con, sql):
    return con.execute(sql).fetchone()[0]


def test_nobody_swipes_on_the_same_person_twice(con):
    assert one(con, "select count(*) from (select viewer_id, target_id from swipes group by all having count(*) > 1)") == 0


def test_nobody_swipes_on_themselves(con):
    assert one(con, "select count(*) from swipes where viewer_id = target_id") == 0


def test_deck_is_same_city_only(con):
    assert one(con, """select count(*) from swipes s
        join profiles v on v.user_id = s.viewer_id join profiles t on t.user_id = s.target_id
        where v.city <> t.city""") == 0


def test_every_like_is_in_liked_profiles(con):
    assert one(con, """select count(*) from swipes s anti join liked_profiles l
        on l.user_id = s.viewer_id and l.liked_user_id = s.target_id where s.direction = 'like'""") == 0


def test_conversations_have_exactly_two_people_who_liked_each_other(con):
    assert one(con, "select count(*) from (select conversation_id from conversation_participants group by 1 having count(*) <> 2)") == 0
    assert one(con, """select count(*) from (
        select conversation_id, min(user_id) a, max(user_id) b from conversation_participants group by 1) c
        where not exists (select 1 from liked_profiles where user_id = c.a and liked_user_id = c.b)
           or not exists (select 1 from liked_profiles where user_id = c.b and liked_user_id = c.a)""") == 0


def test_messages_carry_no_content_column(con):
    cols = [r[0] for r in con.execute("describe messages").fetchall()]
    assert "content" not in cols


def test_score_is_in_display_range(con):
    lo, hi = con.execute("select min(match_percent), max(match_percent) from swipes").fetchone()
    assert lo >= 15
    # Production bug kept on purpose: a move-in bonus can push the score past 100.
    assert hi <= 108
