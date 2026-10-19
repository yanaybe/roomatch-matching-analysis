-- Q1. Does the "NN% match" users see predict anything?
--
-- Three outcomes, each with its honest denominator:
--   like rate      = likes / swipes                      (every card shown)
--   mutual rate    = mutual likes / pairs who BOTH saw each other
--                    (a pair where one side never saw the other can't be mutual,
--                     and counting it would blame the score for the deck order)
--   two-way chat   = conversations where both people sent >= 2 messages / mutual pairs
-- The score is symmetric (score(a,b) = score(b,a)), so a pair has one band.
with banded as (
    select *,
        case when match_percent <= 15 then '15 (floor)'
             when match_percent < 30  then '16-29'
             when match_percent < 45  then '30-44'
             when match_percent < 60  then '45-59'
             when match_percent < 75  then '60-74'
             else '75+' end as band
    from swipes
),
pairs as (   -- one row per unordered pair where both sides swiped
    select a.band,
           a.direction = 'like' and b.direction = 'like' as mutual,
           least(a.viewer_id, a.target_id)    as u1,
           greatest(a.viewer_id, a.target_id) as u2
    from banded a
    join banded b on b.viewer_id = a.target_id and b.target_id = a.viewer_id
    where a.viewer_id < a.target_id
),
per_sender as (
    select conversation_id, sender_id, count(*) as sent
    from messages group by 1, 2
),
chat as (    -- conversations where both participants sent at least 2 messages
    select cp.conversation_id, min(cp.user_id) as u1, max(cp.user_id) as u2
    from conversation_participants cp
    where cp.conversation_id in (
        select conversation_id from per_sender where sent >= 2
        group by 1 having count(*) = 2)
    group by 1
)
select
    s.band,
    s.swipes, s.likes,
    p.pairs_both_saw, p.mutual_pairs,
    coalesce(p.two_way_chats, 0) as two_way_chats
from (select band, count(*) as swipes, count(*) filter (where direction = 'like') as likes
      from banded group by 1) s
left join (
    select pairs.band,
           count(*)                                   as pairs_both_saw,
           count(*) filter (where mutual)             as mutual_pairs,
           count(chat.conversation_id)                as two_way_chats
    from pairs
    left join chat on chat.u1 = pairs.u1 and chat.u2 = pairs.u2 and pairs.mutual
    group by 1
) p on p.band = s.band
order by s.band;
