#!/usr/bin/env python3
"""根据第一步产出的 Hive 边表生成随机游走序列，并写入 Hive。"""

import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import types as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weighted random walks")
    parser.add_argument("--input-table", required=True, help="01 脚本输出边表（Hive，字段: id1,id2,weight）")
    parser.add_argument("--output-table", required=True, help="随机游走输出表（Hive）")
    parser.add_argument("--partitions", required=True, help="Hive 分区表达式，如 dt='20250101'")
    parser.add_argument("--num-walks", type=int, default=10, help="每个点的游走次数")
    parser.add_argument("--walk-length", type=int, default=20, help="单次游走最大长度")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


@F.udf(returnType=T.StringType())
def weighted_pick(neighbors):
    """从邻居列表中按权重采样下一个节点。"""
    if not neighbors:
        return None

    import random

    total = 0.0
    for n in neighbors:
        w = float(n["weight"]) if n["weight"] is not None else 0.0
        if w > 0:
            total += w

    if total <= 0:
        return random.choice(neighbors)["dst"]

    r = random.random() * total
    s = 0.0
    for n in neighbors:
        w = float(n["weight"]) if n["weight"] is not None else 0.0
        if w <= 0:
            continue
        s += w
        if s >= r:
            return n["dst"]
    return neighbors[-1]["dst"]


def main() -> None:
    args = parse_args()

    spark = SparkSession.builder.appName("node2vec_generate_walks").enableHiveSupport().getOrCreate()

    edges = (
        spark.table(args.input_table)
        .select(
            F.col("id1").cast("string").alias("src"),
            F.col("id2").cast("string").alias("dst"),
            F.col("weight").cast("double").alias("weight"),
        )
        .where(F.col("src").isNotNull() & F.col("dst").isNotNull() & F.col("weight").isNotNull())
    )

    directed_edges = edges.unionByName(
        edges.select(F.col("dst").alias("src"), F.col("src").alias("dst"), F.col("weight"))
    )

    neighbors_df = (
        directed_edges.groupBy("src")
        .agg(F.collect_list(F.struct(F.col("dst"), F.col("weight"))).alias("neighbors"))
        .cache()
    )

    nodes_df = neighbors_df.select(F.col("src").alias("start_node")).distinct()

    walks_df = (
        nodes_df.withColumn("walk_idx", F.explode(F.sequence(F.lit(1), F.lit(args.num_walks))))
        .withColumn("walk_id", F.concat_ws("#", F.col("start_node"), F.col("walk_idx")))
        .withColumn("current", F.col("start_node"))
        .withColumn("words", F.array(F.col("start_node")))
        .withColumn("ended", F.lit(False))
        .select("walk_id", "current", "words", "ended")
    )

    for _ in range(args.walk_length - 1):
        walks_df = (
            walks_df.join(neighbors_df, walks_df.current == neighbors_df.src, "left")
            .withColumn(
                "neighbors",
                F.when(F.col("ended"), F.array().cast("array<struct<dst:string,weight:double>>")).otherwise(
                    F.col("neighbors")
                ),
            )
            .withColumn(
                "neighbors",
                F.when(
                    F.col("neighbors").isNull(), F.array().cast("array<struct<dst:string,weight:double>>")
                ).otherwise(F.col("neighbors")),
            )
            .withColumn("neighbors", F.expr("shuffle(neighbors)"))
            .withColumn("next_node", F.when(F.size("neighbors") > 0, weighted_pick("neighbors")))
            .withColumn("ended", F.col("ended") | F.col("next_node").isNull())
            .withColumn("current", F.coalesce(F.col("next_node"), F.col("current")))
            .withColumn(
                "words",
                F.when(F.col("next_node").isNotNull(), F.concat(F.col("words"), F.array(F.col("next_node")))).otherwise(
                    F.col("words")
                ),
            )
            .select("walk_id", "current", "words", "ended")
        )

    result_df = walks_df.select("words")
    result_df.createOrReplaceTempView("result_view")
    insert_sql = "INSERT OVERWRITE TABLE {} PARTITION ({}) select words from result_view".format(
        args.output_table, args.partitions
    )
    spark.sql(insert_sql)

    spark.stop()


if __name__ == "__main__":
    main()
