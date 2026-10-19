-- Q3b. The deck is ordered newest-first. How fast does a profile stop being shown?
-- Impressions = times the profile appeared as a swiped card, by weeks since it joined.
-- Only profiles with at least 8 full weeks of history are included (right-censoring).
select
    cast(floor(date_diff('day', cast(p.created_at as date), cast(s.swiped_at as date)) / 7) as integer) as weeks_since_join,
    count(*)                                         as impressions,
    count(distinct p.user_id)                        as profiles_shown,
    (select count(*) from profiles
     where cast(created_at as date) <= (select max(cast(swiped_at as date)) from swipes) - interval 56 day) as eligible_profiles
from swipes s
join profiles p on p.user_id = s.target_id
where cast(p.created_at as date) <= (select max(cast(swiped_at as date)) from swipes) - interval 56 day
group by 1
having weeks_since_join between 0 and 7
order by 1;
