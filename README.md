# OGV Node2Vec Recall (PySpark)

该目录提供一个基于 **PySpark** 的 Node2Vec 召回离线流程。
按你的反馈，流程改为 **全链路读写 Hive 表**（不再依赖 parquet 落地）。

## 流程拆分

1. `scripts/01_prepare_sequences.py`
   - 直接读取并执行 `gen_samples.sql`
   - 按 `--dt` 替换 SQL 中的日期占位符
   - SQL 内部负责写入 Hive（你当前版本的 SQL 已写 Hive）
2. `scripts/02_build_edges.py`
   - 读取第一步产出的 Hive 表（字段 `id1`, `id2`, `weight`）
   - 清洗后写入标准化边表（`src`, `dst`, `weight`）
3. `scripts/03_generate_walks.py`
   - 读取边 Hive 表，生成加权随机游走序列并写 Hive
4. `scripts/04_train_embeddings.py`
   - 读取游走 Hive 表，训练 `Word2Vec`，写 embedding Hive 表
5. `scripts/05_build_recall.py`
   - 读取 embedding Hive 表，计算 TopK 召回并写 Hive

## 典型执行示例

```bash
spark-submit scripts/01_prepare_sequences.py \
  --dt 20250101 \
  --sql gen_samples.sql

spark-submit scripts/02_build_edges.py \
  --input-table ai.tmp_node2vec_edge_raw_20250101 \
  --output-table ai.tmp_node2vec_edge_20250101

spark-submit scripts/03_generate_walks.py \
  --input-table ai.tmp_node2vec_edge_20250101 \
  --output-table ai.tmp_node2vec_walk_20250101 \
  --num-walks 10 \
  --walk-length 20

spark-submit scripts/04_train_embeddings.py \
  --input-table ai.tmp_node2vec_walk_20250101 \
  --output-table ai.tmp_node2vec_emb_20250101

spark-submit scripts/05_build_recall.py \
  --input-table ai.tmp_node2vec_emb_20250101 \
  --output-table ai.tmp_node2vec_recall_20250101 \
  --topk 100
```

> 说明：1~5 步都依赖 Hive 元数据权限，请确保 SparkSession 已启用 Hive 支持。
