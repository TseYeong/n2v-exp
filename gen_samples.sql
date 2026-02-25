WITH
  season_avids AS (
    SELECT DISTINCT
      season_id,
      avid
    FROM
      bili_ogv.dim_ep_av_full_d
    WHERE
      ep_delete_status = 0
      AND season_delete_status = 0
      AND format_delete_status = 0
      AND is_published = 1
      AND is_huaxu = 0
      AND log_date = '${SCHEDULE_TIME, yyyyMMdd, -1d}'
  ),
  av_play_info AS (
    SELECT
      mid,
      avid,
      CAST(
        CAST(
          unix_timestamp(
            CONCAT(
              substr(log_date, 1, 4),
              '-',
              substr(log_date, 5, 2),
              '-',
              substr(log_date, 7, 2),
              ' 00:00:00'
            )
          ) AS BIGINT
        ) + click_time AS INT
      ) AS click_time,
      duration_sum
    FROM
      ai.recsys_action_instance_fix
    WHERE
      log_date >= '${SCHEDULE_TIME, yyyMMdd, -7d}'
      AND duration_sum > 5
      AND mid > 0
      AND avid > 0
  ),
  played_avid AS (
    SELECT
      avid
    FROM
      av_play_info
    GROUP BY
      1
    HAVING
      COUNT(avid) >= 1000
  ),
  valid_av AS (
    SELECT
      avid
    FROM
      (
        SELECT
          tb1.avid,
          row_number() OVER (
            ORDER BY
              cnt DESC
          ) AS rank
        FROM
          (
            SELECT
              t0.avid,
              t1.play - COALESCE(t2.play, 0) AS cnt
            FROM
              played_avid t0
              JOIN (
                SELECT
                  CAST(avid AS BIGINT) AS avid,
                  CAST(play AS BIGINT) AS play
                FROM
                  ai.recsys_avbasic_dist
                WHERE
                  log_date = '${SCHEDULE_TIME, yyyyMMdd, -1d}'
                  AND log_hour = '23'
              ) t1 ON t0.avid = t1.avid
              LEFT JOIN (
                SELECT
                  CAST(avid AS BIGINT) AS avid,
                  CAST(play AS BIGINT) AS play
                FROM
                  ai.recsys_avbasic_dist
                WHERE
                  log_date = '${SCHEDULE_TIME, yyyyMMdd, -5d}'
                  AND log_hour = '23'
              ) t2 ON t1.avid = t2.avid
          ) tb1
      ) tb2
    WHERE
      rank <= 500000
  ),
  av_sample AS (
    SELECT
      a.mid,
      CASE
        WHEN e.season_id IS NOT NULL THEN CONCAT('ss_', CAST(e.season_id AS STRING))
        ELSE CAST(a.avid AS STRING)
      END AS itemid,
      a.click_time
    FROM
      av_play_info a
      JOIN valid_av b ON a.avid = b.avid
      LEFT JOIN season_avids e ON a.avid = e.avid
    WHERE
      e.season_id IS NOT NULL
      AND duration_sum > 90
      OR e.season_id IS NULL
  ),
  season_played_user AS (
    SELECT
      mid
    FROM
      av_sample
    WHERE
      itemid LIKE 'ss%'
    GROUP BY
      1
  ),
  sample AS (
    SELECT
      t1.mid,
      itemid,
      click_time
    FROM
      av_sample t1
      JOIN season_played_user t2 ON t1.mid = t2.mid
  ),
  ordered_sample AS (
    SELECT
      mid,
      itemid,
      rank
    FROM
      (
        SELECT
          mid,
          itemid,
          ROW_NUMBER() OVER (
            PARTITION BY
              mid
            ORDER BY
              click_time DESC
          ) AS rank
        FROM
          sample
      )
    WHERE
      rank <= 300
  ),
  pair AS (
    SELECT
      t1.mid AS mid,
      t1.itemid AS itemid1,
      t2.itemid AS itemid2
    FROM
      (
        SELECT
          mid,
          itemid,
          rank - 1 AS rank
        FROM
          ordered_sample
      ) t1
      JOIN ordered_sample t2 ON t1.mid = t2.mid
      AND t1.rank = t2.rank
      AND t1.itemid <> t2.itemid
  )
INSERT OVERWRITE ai.ogv_node2vec_edge_d PARTITION (log_date = '${SCHEDULE_TIME, yyyyMMdd, -1d}')
SELECT
  id1,
  id2,
  ln(cnt) AS weight
FROM
  (
    SELECT
      id1,
      id2,
      COUNT(mid) AS cnt
    FROM
      (
        SELECT
          mid,
          IF (itemid1 < itemid2, itemid1, itemid2) id1,
          IF (itemid1 > itemid2, itemid1, itemid2) id2
        FROM
          pair
      )
    GROUP BY
      1,
      2
  )
WHERE
  cnt > 1
