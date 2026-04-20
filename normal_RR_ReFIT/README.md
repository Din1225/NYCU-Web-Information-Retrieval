# Dense-only ReFIT

這個資料夾實作 ReFIT 的核心流程，不混合 BM25：

1. 用 dense retriever 對 query 做第一次檢索，取得 feedback top-K。
2. 用 reranker 對同一批 dense candidates 計算 relevance feedback 分數。
3. 將 reranker 分數做 min-max normalization，再以 temperature `T=2` 轉成 softmax 分布。
4. 將 retriever 內積分數做 min-max normalization，再轉成 softmax 分布。
5. 用 KL divergence 更新 query embedding，預設 `n=100`、learning rate `0.005`。
6. 用更新後的 query embedding 做第二次 dense retrieval，第二次結果不再 rerank。

## Debug run

```bash
python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/debug_refit_100docs_results.json \
  --limit_docs 100 \
  --limit_queries 1 \
  --feedback_top_k 50 \
  --final_top_k 30 \
  --refit_updates 10 \
  --use_fp16 \
  --log_file outputs/logs/debug_refit_100docs.log \
  --cuda_visible_devices 3
```

## Full run

```bash
python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/refit_results.json \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --use_fp16 \
  --log_file outputs/logs/refit.log \
  --cuda_visible_devices 3
```

若要指定 GPU，可以加上：

```bash
--cuda_visible_devices 3
```

