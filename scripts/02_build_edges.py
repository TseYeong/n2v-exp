#!/usr/bin/env python3
"""将用户序列转为 item-item 无向加权边。"""

import argparse
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weighted co-occurrence edges")
    parser.add_argument("--input", required=True, help="01 脚本输出路径")
    parser.add_argument("--output", required=True, help="边表输出路径（parquet）")
    parser.add_argument("--min-count", type=int, default=2, help="最小共现次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_build_edges").getOrCreate()

    seq_df = spark.read.parquet(args.input)

    pair_df = (
        seq_df.select(
            "mid",
            F.posexplode(F.col("items")).alias("idx", "itemid1"),
            F.col("items"),
        )
        .where(F.col("idx") < F.size("items") - 1)
        .select(
            "mid",
            "itemid1",
            F.col("items")[F.col("idx") + 1].alias("itemid2"),
        )
        .where(F.col("itemid1") != F.col("itemid2"))
    )

    edge_df = (
        pair_df.select(
            "mid",
            F.when(F.col("itemid1") < F.col("itemid2"), F.col("itemid1"))
            .otherwise(F.col("itemid2"))
            .alias("src"),
            F.when(F.col("itemid1") > F.col("itemid2"), F.col("itemid1"))
            .otherwise(F.col("itemid2"))
            .alias("dst"),
        )
        .groupBy("src", "dst")
        .agg(F.count("mid").alias("cnt"))
        .where(F.col("cnt") >= args.min_count)
        .select("src", "dst", F.log(F.col("cnt")).alias("weight"))
    )

    edge_df.write.mode("overwrite").parquet(args.output)

    spark.stop()


if __name__ == "__main__":
    main()
