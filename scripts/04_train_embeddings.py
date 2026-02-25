#!/usr/bin/env python3
"""在随机游走序列上训练 item embedding，结果写入 Hive。"""

import argparse
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train item embeddings with Spark Word2Vec")
    parser.add_argument("--input-table", required=True, help="03 脚本输出表（Hive）")
    parser.add_argument("--output-table", required=True, help="embedding 输出表（Hive）")
    parser.add_argument("--partitions", required=True, help="Hive 分区表达式，如 dt='20250101'")
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_train_embeddings").enableHiveSupport().getOrCreate()

    walk_df = spark.table(args.input_table)
    model = Word2Vec(
        vectorSize=args.vector_size,
        windowSize=args.window_size,
        minCount=args.min_count,
        maxIter=args.max_iter,
        inputCol="words",
        outputCol="embedding",
    ).fit(walk_df)

    result_df = model.getVectors().select("word", "vector")
    result_df.createOrReplaceTempView("result_view")
    insert_sql = "INSERT OVERWRITE TABLE {} PARTITION ({}) select word, vector from result_view".format(
        args.output_table, args.partitions
    )
    spark.sql(insert_sql)

    spark.stop()


if __name__ == "__main__":
    main()
