#!/usr/bin/env python3
"""读取第一步 SQL 输出，转成结构化边表供后续步骤使用。"""

import argparse
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load step1 edge text output and save as parquet")
    parser.add_argument("--input", required=True, help="01 脚本输出目录（text）")
    parser.add_argument("--output", required=True, help="结构化边表输出目录（parquet）")
    parser.add_argument("--min-weight", type=float, default=0.0, help="最小边权重过滤")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_load_edges_from_sql_output").getOrCreate()

    # SQL 输出格式: id1 id2 weight（空格分隔）
    text_df = spark.read.text(args.input)
    edge_df = (
        text_df.select(F.split(F.col("value"), r"\\s+").alias("parts"))
        .where(F.size("parts") >= 3)
        .select(
            F.col("parts")[0].alias("src"),
            F.col("parts")[1].alias("dst"),
            F.col("parts")[2].cast("double").alias("weight"),
        )
        .where(F.col("src").isNotNull() & F.col("dst").isNotNull() & F.col("weight").isNotNull())
        .where(F.col("weight") >= F.lit(args.min_weight))
    )

    edge_df.write.mode("overwrite").parquet(args.output)
    spark.stop()


if __name__ == "__main__":
    main()
