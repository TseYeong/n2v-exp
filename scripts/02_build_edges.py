#!/usr/bin/env python3
"""读取第一步写入的 Hive 边表，清洗后写入下游 Hive 边表。"""

import argparse
from pyspark.sql import SparkSession, functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read step1 Hive table and write cleaned edge Hive table")
    parser.add_argument("--input-table", required=True, help="01 步产出的 Hive 表")
    parser.add_argument("--output-table", required=True, help="02 步输出 Hive 表")
    parser.add_argument("--min-weight", type=float, default=0.0, help="最小边权重过滤")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_load_edges_from_hive").enableHiveSupport().getOrCreate()

    edge_df = (
        spark.table(args.input_table)
        .select(
            F.col("id1").cast("string").alias("src"),
            F.col("id2").cast("string").alias("dst"),
            F.col("weight").cast("double").alias("weight"),
        )
        .where(F.col("src").isNotNull() & F.col("dst").isNotNull() & F.col("weight").isNotNull())
        .where(F.col("weight") >= F.lit(args.min_weight))
    )

    edge_df.write.mode("overwrite").saveAsTable(args.output_table)
    spark.stop()


if __name__ == "__main__":
    main()
