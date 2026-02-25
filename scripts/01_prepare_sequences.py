#!/usr/bin/env python3
"""根据既有 SQL 逻辑生成 Node2Vec 训练序列。"""

import argparse
from pyspark.sql import SparkSession


SQL_TEMPLATE = """
WITH
  season_avids AS (
    SELECT DISTINCT season_id, avid
    FROM bili_ogv.dim_ep_av_full_d
    WHERE ep_delete_status = 0
      AND season_delete_status = 0
      AND format_delete_status = 0
      AND is_published = 1
      AND is_huaxu = 0
      AND log_date = '{dt_m1}'
  ),
  av_play_info AS (
    SELECT
      mid,
      avid,
      CAST(
        CAST(
          unix_timestamp(
            CONCAT(
              substr(log_date, 1, 4), '-', substr(log_date, 5, 2), '-', substr(log_date, 7, 2), ' 00:00:00'
            )
          ) AS BIGINT
        ) + click_time AS INT
      ) AS click_time,
      duration_sum
    FROM ai.recsys_action_instance_fix
    WHERE log_date >= '{dt_m7}'
      AND duration_sum > 5
      AND mid > 0
      AND avid > 0
  ),
  played_avid AS (
    SELECT avid
    FROM av_play_info
    GROUP BY avid
    HAVING COUNT(avid) >= 1000
  ),
  valid_av AS (
    SELECT avid
    FROM (
      SELECT
        tb1.avid,
        row_number() OVER (ORDER BY cnt DESC) AS rank
      FROM (
        SELECT
          t0.avid,
          t1.play - COALESCE(t2.play, 0) AS cnt
        FROM played_avid t0
        JOIN (
          SELECT CAST(avid AS BIGINT) AS avid, CAST(play AS BIGINT) AS play
          FROM ai.recsys_avbasic_dist
          WHERE log_date = '{dt_m1}' AND log_hour = '23'
        ) t1 ON t0.avid = t1.avid
        LEFT JOIN (
          SELECT CAST(avid AS BIGINT) AS avid, CAST(play AS BIGINT) AS play
          FROM ai.recsys_avbasic_dist
          WHERE log_date = '{dt_m5}' AND log_hour = '23'
        ) t2 ON t1.avid = t2.avid
      ) tb1
    ) tb2
    WHERE rank <= {topn}
  ),
  av_sample AS (
    SELECT
      a.mid,
      CASE WHEN e.season_id IS NOT NULL THEN CONCAT('ss_', CAST(e.season_id AS STRING))
           ELSE CAST(a.avid AS STRING)
      END AS itemid,
      a.click_time
    FROM av_play_info a
    JOIN valid_av b ON a.avid = b.avid
    LEFT JOIN season_avids e ON a.avid = e.avid
    WHERE (e.season_id IS NOT NULL AND duration_sum > 90)
       OR e.season_id IS NULL
  ),
  season_played_user AS (
    SELECT mid
    FROM av_sample
    WHERE itemid LIKE 'ss%'
    GROUP BY mid
  ),
  sample AS (
    SELECT t1.mid, itemid, click_time
    FROM av_sample t1
    JOIN season_played_user t2 ON t1.mid = t2.mid
  ),
  ordered_sample AS (
    SELECT
      mid,
      itemid,
      ROW_NUMBER() OVER (PARTITION BY mid ORDER BY click_time DESC) AS rank
    FROM sample
  )
SELECT
  mid,
  transform(
    sort_array(collect_list(named_struct('rank', rank, 'itemid', itemid))),
    x -> x.itemid
  ) AS items
FROM ordered_sample
WHERE rank <= {max_seq_len}
GROUP BY mid
HAVING size(items) >= {min_seq_len}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare user item sequences for node2vec")
    parser.add_argument("--dt", required=True, help="调度日期，格式 yyyyMMdd")
    parser.add_argument("--output", required=True, help="输出路径（parquet）")
    parser.add_argument("--topn", type=int, default=500000, help="有效 avid 热度截断")
    parser.add_argument("--max-seq-len", type=int, default=300, help="每个用户最大序列长度")
    parser.add_argument("--min-seq-len", type=int, default=2, help="保留的最短序列长度")
    return parser.parse_args()


def shift_yyyymmdd(dt: str, days: int) -> str:
    from datetime import datetime, timedelta

    base = datetime.strptime(dt, "%Y%m%d")
    return (base + timedelta(days=days)).strftime("%Y%m%d")


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder.appName("node2vec_prepare_sequences")
        .enableHiveSupport()
        .getOrCreate()
    )

    sql = SQL_TEMPLATE.format(
        dt_m1=shift_yyyymmdd(args.dt, -1),
        dt_m5=shift_yyyymmdd(args.dt, -5),
        dt_m7=shift_yyyymmdd(args.dt, -7),
        topn=args.topn,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
    )

    seq_df = spark.sql(sql)
    seq_df.write.mode("overwrite").parquet(args.output)

    spark.stop()


if __name__ == "__main__":
    main()
