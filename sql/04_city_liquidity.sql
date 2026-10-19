-- Q4. Is the marketplace liquid everywhere? Per city: pool size, and the share of
-- new users who get at least one mutual like within 30 days of joining.
-- Only users who joined 30+ days before the data ends are counted.
with mutual as (
    select a.viewer_id as user_id,
           greatest(a.swiped_at, b.swiped_at) as matched_at
    from swipes a
    join swipes b on b.viewer_id = a.target_id and b.target_id = a.viewer_id
    where a.direction = 'like' and b.direction = 'like'
),
eligible as (
    select user_id, city, cast(created_at as date) as joined
    from profiles
    where cast(created_at as date) <= (select max(cast(swiped_at as date)) from swipes) - interval 30 day
)
select
    e.city,
    count(*)                                                       as new_users,
    count(*) filter (where exists (
        select 1 from mutual m
        where m.user_id = e.user_id
          and cast(m.matched_at as date) < e.joined + interval 30 day)) as matched_within_30d,
    (select count(*) from profiles p where p.city = e.city)       as city_pool,
    (select median(budget) from profiles p where p.city = e.city) as median_budget
from eligible e
group by 1
order by new_users desc;
