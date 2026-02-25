#!/usr/bin/env python3
"""根据 embedding 构建 item topK 召回对，并写入 Hive。"""

import argparse
import math

from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topK recall candidates from embeddings")
    parser.add_argument("--input-table", required=True, help="04 脚本输出表（Hive）")
    parser.add_argument("--output-table", required=True, help="召回结果输出表（Hive）")
    parser.add_argument("--partitions", required=True, help="Hive 分区表达式，如 dt='20250101'")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--min-sim", type=float, default=0.2)
    parser.add_argument("--num-hash-tables", type=int, default=4, help="LSH 哈希表数")
    parser.add_argument("--bucket-length", type=float, default=1.5, help="LSH 桶长度")
    return parser.parse_args()


def cosine_to_l2_threshold(min_sim: float) -> float:
    """归一化向量下: cos = 1 - d^2/2  => d = sqrt(2 - 2*cos)."""
    sim = max(-1.0, min(1.0, min_sim))
    return math.sqrt(max(0.0, 2.0 - 2.0 * sim))


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_build_recall").enableHiveSupport().getOrCreate()

    emb_df = spark.table(args.input_table).select(
        F.col("word").alias("itemid"),
        F.col("vector").cast(T.ArrayType(T.DoubleType())).alias("vector"),
    )

    vec_df = emb_df.select("itemid", F.array_to_vector("vector").alias("features"))
    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
    norm_df = normalizer.transform(vec_df).select("itemid", F.col("norm_features"))

    lsh = BucketedRandomProjectionLSH(
        inputCol="norm_features",
        outputCol="hashes",
        bucketLength=args.bucket_length,
        numHashTables=args.num_hash_tables,
    )
    lsh_model = lsh.fit(norm_df)

    max_l2_dist = cosine_to_l2_threshold(args.min_sim)
    pair_df = lsh_model.approxSimilarityJoin(
        norm_df.alias("l"),
        norm_df.alias("r"),
        threshold=max_l2_dist,
        distCol="dist",
    ).select(
        F.col("datasetA.itemid").alias("src_item"),
        F.col("datasetB.itemid").alias("dst_item"),
        F.col("dist"),
    )

    pair_df = (
        pair_df.where(F.col("src_item") != F.col("dst_item"))
        .withColumn("score", F.lit(1.0) - (F.col("dist") * F.col("dist")) / F.lit(2.0))
        .where(F.col("score") >= F.lit(args.min_sim))
        .select("src_item", "dst_item", "score")
    )

    window = Window.partitionBy("src_item").orderBy(F.col("score").desc())
    recall_df = (
        pair_df.withColumn("rk", F.row_number().over(window))
        .where(F.col("rk") <= args.topk)
        .drop("rk")
    )

    recall_df.createOrReplaceTempView("result_view")
    insert_sql = "INSERT OVERWRITE TABLE {} PARTITION ({}) select src_item, dst_item, score from result_view".format(
        args.output_table, args.partitions
    )
    spark.sql(insert_sql)

    spark.stop()


if __name__ == "__main__":
    main()
