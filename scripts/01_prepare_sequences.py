#!/usr/bin/env python3
"""直接执行仓库中的 gen_samples.sql，产出 node2vec 边样本。"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession


PLACEHOLDER_PATTERNS = {
    "${SCHEDULE_TIME, yyyyMMdd, -1d}": -1,
    "${SCHEDULE_TIME, yyyyMMdd, -5d}": -5,
    "${SCHEDULE_TIME, yyyyMMdd, -7d}": -7,
    "${SCHEDULE_TIME, yyyMMdd, -1d}": -1,
    "${SCHEDULE_TIME, yyyMMdd, -5d}": -5,
    "${SCHEDULE_TIME, yyyMMdd, -7d}": -7,
}

DEFAULT_OUTPUT_PLACEHOLDER = (
    "'/department/ai/user/xieyun/online_data/ogv_recall/ogv_node2vec_v1/edge/"
    "${SCHEDULE_TIME, yyyyMMdd, -1d}'"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provided SQL directly to generate node2vec edge samples")
    parser.add_argument("--dt", required=True, help="调度日期，格式 yyyyMMdd")
    parser.add_argument("--sql", default="gen_samples.sql", help="SQL 文件路径")
    parser.add_argument("--output", required=True, help="第一步输出目录（text，空格分隔）")
    return parser.parse_args()


def shift_yyyymmdd(dt: str, days: int) -> str:
    base = datetime.strptime(dt, "%Y%m%d")
    return (base + timedelta(days=days)).strftime("%Y%m%d")


def render_sql(sql_text: str, dt: str, output_path: str) -> str:
    rendered = sql_text
    for token, delta in PLACEHOLDER_PATTERNS.items():
        rendered = rendered.replace(token, shift_yyyymmdd(dt, delta))

    # 覆盖 INSERT OVERWRITE directory 的默认路径，确保由参数控制输出。
    output_literal = f"'{output_path}'"
    rendered = rendered.replace(DEFAULT_OUTPUT_PLACEHOLDER, output_literal)
    return rendered


def main() -> None:
    args = parse_args()

    sql_path = Path(args.sql)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL 文件不存在: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    final_sql = render_sql(sql_text, args.dt, args.output)

    spark = SparkSession.builder.appName("node2vec_run_sample_sql").enableHiveSupport().getOrCreate()
    spark.sql(final_sql)
    spark.stop()


if __name__ == "__main__":
    main()
