# 實驗

這個資料夾現在提供一條獨立 baseline，只做：

1. dense retrieval
2. reranker rerank
3. 用 `114B-WIR-Phase1-Annotation.json` 做 masked internal evaluation

不做 ReFIT，不改 query embedding，也不混入 BM25。

## 實驗目的

第一階段先看最基本的 `Retrieve -> Rerank` 效果，避免 query 向量更新把結果搞亂。若 baseline 就已經不理想，再往後調整方法。

目前支援的 retriever / reranker 組合：

- `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`
- `BAAI/bge-m3` + `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Embedding-4B` + `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-Embedding-4B` + `Qwen/Qwen3-Reranker-4B`

## 入口

- 執行實驗：`exp/scripts/run_dense_rerank.py`
- 單獨評估：`exp/scripts/evaluate_dense_rerank.py`

## 輸出內容

每個 query 會同時保留兩份結果：

- `dense_results`：只看 dense retriever 的 top-K
- `rerank_results`：對 `dense_results` 做 rerank 後的 top-K

所以你可以直接比較：

- 只有 dense 的效果
- dense + rerank 的效果

## 評估規則

評估使用 `114B-WIR-Phase1-Annotation.json`。

- phase1 的 10 個 query 都有部分標註
- label 使用 `final_answer`，範圍是 `0 / 1 / 2`
- 若檢索結果中的文件沒有被標註，會先從排名中移除，不列入算分
- 再對剩下的已標註文件計算：
  - `MAP@10`
  - `MAP@30`
  - `MRR`
  - `NDCG@10`
  - `NDCG@30`

預設情況下：

- `MAP` / `MRR` 會把 `label >= 2` 視為 relevant
- `NDCG` 直接使用 `0 / 1 / 2` 的 graded relevance

若你想改成比較寬鬆，把 `1` 和 `2` 都當 relevant，可以加：

```bash
--binary_relevance_threshold 1
```

## 預設文件表示

預設 `document_text_mode=question`，也就是：

- dense retriever 只看 `Question`
- reranker 也只看 `Question`

這比較接近你目前要測的「普通 dense retriever + reranker」baseline。

如果想另外測 `Question + Answer`，可以改成：

```bash
--document_text_mode question_answer
```

## 範例

### 1. BGE + BGE

```bash
python exp/scripts/run_dense_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/bge_bge_phase1_results.json \
  --dense_model BAAI/bge-m3 \
  --reranker_model BAAI/bge-reranker-v2-m3 \
  --dense_top_k 100 \
  --rerank_top_k 30 \
  --bge_use_fp16 \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --evaluation_output outputs/bge_bge_phase1_evaluation.json
```

### 2. BGE + Qwen

```bash
python exp/scripts/run_dense_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/bge_qwen_phase1_results.json \
  --dense_model BAAI/bge-m3 \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --dense_top_k 100 \
  --rerank_top_k 30 \
  --bge_use_fp16 \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --evaluation_output outputs/bge_qwen_phase1_evaluation.json \
  --reranker_batch_size 2 \
  --cuda_visible_devices 0
```

### 3. Qwen + BGE

```bash
python exp/scripts/run_dense_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_bge_phase1_results.json \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model BAAI/bge-reranker-v2-m3 \
  --dense_top_k 100 \
  --rerank_top_k 30 \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --evaluation_output outputs/qwen_bge_phase1_evaluation.json \
  --cuda_visible_devices 0
```

### 4. Qwen + Qwen

```bash
python exp/scripts/run_dense_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_qwen_phase1_results.json \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --dense_top_k 100 \
  --rerank_top_k 30 \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --reranker_batch_size 2 \
  --evaluation_output outputs/qwen_qwen_phase1_evaluation.json \
  --cuda_visible_devices 0
```

## 單獨重跑評估

如果結果 JSON 已經存在，可以只重跑評估：

```bash
python exp/scripts/evaluate_dense_rerank.py \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --result exp/outputs/hybrid_union_query1_results_for_eval.json \
  --result_field both \
  --output exp/outputs/hybrid_union_query1_results_for_eval.json


python exp/scripts/evaluate_dense_rerank.py \
  --annotation 114B-WIR-Phase1-Annotation.json \
  --result exp/outputs/hybrid_union_query1_LLMrisk_results_for_eval3.json \
  --result_field rerank_results
```

## 重要參數

- `--dense_top_k`：dense retrieval 先取幾篇文件交給 reranker
- `--rerank_top_k`：最後保留幾筆 rerank 結果
- `--dense_backend` / `--reranker_backend`：若 model name 無法自動判斷，可手動指定 `bge` 或 `qwen`
- `--document_text_mode`：`question` 或 `question_answer`
- `--bge_use_fp16`：BGE backend 是否用 fp16
- `--disable_4bit`：關閉 Qwen 4-bit 量化
- `--binary_relevance_threshold`：控制 MAP / MRR 何時算 relevant

## 注意事項

- 若使用 Hugging Face model id 而本機尚未快取模型，推論框架可能會嘗試下載模型。
- Qwen 4-bit 預設需要 CUDA。
- `FlagEmbedding` 需要能正確載入 BGE 模型。
