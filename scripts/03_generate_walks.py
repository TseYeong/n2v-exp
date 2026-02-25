#!/usr/bin/env python3
"""根据第一步产出的 Hive 边表生成随机游走序列，并写入 Hive。"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession, functions as F


NeighborMap = Dict[str, List[Tuple[str, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weighted random walks")
    parser.add_argument("--input-table", required=True, help="01 脚本输出边表（Hive，字段: id1,id2,weight）")
    parser.add_argument("--output-table", required=True, help="随机游走输出表（Hive）")
    parser.add_argument("--num-walks", type=int, default=10, help="每个点的游走次数")
    parser.add_argument("--walk-length", type=int, default=20, help="单次游走最大长度")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def weighted_pick(neighbors: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in neighbors)
    if total <= 0:
        return random.choice(neighbors)[0]
    r = random.random() * total
    s = 0.0
    for node, w in neighbors:
        s += w
        if s >= r:
            return node
    return neighbors[-1][0]


def build_neighbor_map(edge_rows) -> NeighborMap:
    graph: NeighborMap = defaultdict(list)
    for src, dst, weight in edge_rows:
        w = float(weight)
        graph[src].append((dst, w))
        graph[dst].append((src, w))
    return graph


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

    edge_rows = edges.rdd.map(lambda r: (r[0], r[1], r[2])).collect()
    graph = build_neighbor_map(edge_rows)

    nodes = list(graph.keys())
    sc = spark.sparkContext
    graph_bc = sc.broadcast(graph)

    tasks = [(n, i) for n in nodes for i in range(args.num_walks)]

    def walk_partition(records):
        random.seed(args.seed)
        g = graph_bc.value
        for start, _ in records:
            walk = [start]
            cur = start
            for _ in range(args.walk_length - 1):
                nbrs = g.get(cur, [])
                if not nbrs:
                    break
                nxt = weighted_pick(nbrs)
                walk.append(nxt)
                cur = nxt
            yield (walk,)

    walks_rdd = sc.parallelize(tasks, numSlices=max(1, len(nodes) // 2000 + 1)).mapPartitions(walk_partition)
    walks_df = spark.createDataFrame(walks_rdd, ["words"])
    walks_df.write.mode("overwrite").saveAsTable(args.output_table)

    spark.stop()


if __name__ == "__main__":
    main()
