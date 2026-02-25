# OGV Node2Vec Recall (PySpark)

该目录提供一个基于 **PySpark** 的 Node2Vec 召回离线流程。
按反馈做了拆分：
- **第一步直接执行仓库里已有的 `gen_samples.sql`** 生成边样本；
- **第二步读取第一步输出**，转成结构化边表供后续任务使用。

## 流程拆分

1. `scripts/01_prepare_sequences.py`
   - 直接读取并执行 `gen_samples.sql`
   - 按 `--dt` 替换 SQL 中的日期占位符
   - 将 `INSERT OVERWRITE directory` 输出路径替换为 `--output`
2. `scripts/02_build_edges.py`
   - 读取第一步输出的 text 边文件（`id1 id2 weight`）
   - 转为标准列（`src`, `dst`, `weight`）并写 parquet
3. `scripts/03_generate_walks.py`
   - 基于边表生成加权随机游走序列
4. `scripts/04_train_embeddings.py`
   - 使用 Spark ML `Word2Vec` 在随机游走序列上训练向量
5. `scripts/05_build_recall.py`
   - 基于向量构建近邻召回对（TopK）

## 典型执行示例

```bash
spark-submit scripts/01_prepare_sequences.py \
  --dt 20250101 \
  --sql gen_samples.sql \
  --output /path/node2vec/edge_raw/20250101

spark-submit scripts/02_build_edges.py \
  --input /path/node2vec/edge_raw/20250101 \
  --output /path/node2vec/edge/20250101

spark-submit scripts/03_generate_walks.py \
  --input /path/node2vec/edge/20250101 \
  --output /path/node2vec/walk/20250101 \
  --num-walks 10 \
  --walk-length 20

spark-submit scripts/04_train_embeddings.py \
  --input /path/node2vec/walk/20250101 \
  --output /path/node2vec/emb/20250101

spark-submit scripts/05_build_recall.py \
  --input /path/node2vec/emb/20250101 \
  --output /path/node2vec/recall/20250101 \
  --topk 100
```

> 说明：第一步依赖 Hive 表；请确保 Spark Session 启用 Hive 支持，且有对应表权限。
