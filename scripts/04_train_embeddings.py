#!/usr/bin/env python3
"""在随机游走序列上训练 item embedding。"""

import argparse
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train item embeddings with Spark Word2Vec")
    parser.add_argument("--input", required=True, help="03 脚本输出路径")
    parser.add_argument("--output", required=True, help="embedding 输出路径")
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_train_embeddings").getOrCreate()

    walk_df = spark.read.parquet(args.input)
    model = Word2Vec(
        vectorSize=args.vector_size,
        windowSize=args.window_size,
        minCount=args.min_count,
        maxIter=args.max_iter,
        inputCol="words",
        outputCol="embedding",
    ).fit(walk_df)

    model.getVectors().write.mode("overwrite").parquet(args.output)
    spark.stop()


if __name__ == "__main__":
    main()
