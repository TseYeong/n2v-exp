# OGV Node2Vec Recall (PySpark)

该目录提供一个基于 **PySpark** 的 Node2Vec 召回离线流程，将现有 `gen_samples.sql` 的样本生成逻辑拆分为多个脚本，便于调度和维护。

## 流程拆分

1. `scripts/01_prepare_sequences.py`
   - 复刻 `gen_samples.sql` 的样本过滤逻辑
   - 产出按用户时间倒序的 `item` 序列
2. `scripts/02_build_edges.py`
   - 将用户序列转为 item-item 共现边
   - 输出无向加权图（`src`, `dst`, `weight`）
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
  --output /path/node2vec/seq/20250101

spark-submit scripts/02_build_edges.py \
  --input /path/node2vec/seq/20250101 \
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

> 说明：脚本默认使用 Hive 表；请确保 Spark Session 启用 Hive 支持，且有对应表权限。
