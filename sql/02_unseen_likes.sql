-- Q3a. How many likes can never become a match because the other person
-- never saw the liker? A like only becomes mutual if the target swipes back,
-- and the target can only swipe on someone their deck showed them.
select
    count(*)                                                           as likes,
    count(*) filter (where back.viewer_id is null)                     as target_never_saw_liker,
    count(*) filter (where back.direction = 'like')                    as liked_back,
    count(*) filter (where back.direction = 'pass')                    as passed_back
from swipes l
left join swipes back on back.viewer_id = l.target_id and back.target_id = l.viewer_id
where l.direction = 'like';
