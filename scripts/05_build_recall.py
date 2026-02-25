#!/usr/bin/env python3
"""根据 embedding 构建 item topK 召回对。"""

import argparse
from pyspark.ml.feature import Normalizer
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topK recall candidates from embeddings")
    parser.add_argument("--input", required=True, help="04 脚本输出路径")
    parser.add_argument("--output", required=True, help="召回结果输出路径")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--min-sim", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_build_recall").getOrCreate()

    emb_df = spark.read.parquet(args.input).select(
        F.col("word").alias("itemid"),
        F.col("vector").cast(T.ArrayType(T.DoubleType())).alias("vector"),
    )

    vec_df = emb_df.select("itemid", F.array_to_vector("vector").alias("features"))
    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
    norm_df = normalizer.transform(vec_df).select("itemid", F.col("norm_features"))

    left = norm_df.alias("l")
    right = norm_df.alias("r")

    dot_product = F.expr(
        "aggregate(zip_with(vector_to_array(l.norm_features), vector_to_array(r.norm_features), (x, y) -> x * y), 0D, (acc, x) -> acc + x)"
    )

    pair_df = (
        left.join(right, F.col("l.itemid") != F.col("r.itemid"))
        .select(
            F.col("l.itemid").alias("src_item"),
            F.col("r.itemid").alias("dst_item"),
            dot_product.alias("score"),
        )
        .where(F.col("score") >= args.min_sim)
    )

    window = Window.partitionBy("src_item").orderBy(F.col("score").desc())
    recall_df = (
        pair_df.withColumn("rk", F.row_number().over(window))
        .where(F.col("rk") <= args.topk)
        .drop("rk")
    )

    recall_df.write.mode("overwrite").parquet(args.output)
    spark.stop()


if __name__ == "__main__":
    main()
