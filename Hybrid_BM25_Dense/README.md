# Hybrid BM25 + Dense Fusion Rerank

這個資料夾提供一條和 `normal_RR_GFF_ReFIT` 平行、彼此獨立的新方法線，不會修改原本的 `GFF + ReFIT` 程式。

此方法的核心流程如下：

1. 讀取 `data/IR_data.json` 與 query JSON。
2. 將每篇 document 組成 `Question + Answer` 文字。
3. 建立或讀取 BM25 index。
4. 建立或讀取 dense document embeddings。
5. 對每個 query：
   - 用 BM25 取 top-100 關鍵字候選。
   - 用 dense retrieval 取 top-100 語意候選。
   - 預設將兩邊結果做 union，得到一個可能大於 100 的候選集合。
   - 若傳入 `--RRF`，則改用 Reciprocal Rank Fusion, RRF 先融合 BM25 / dense 排名。
   - 用 reranker 先取 top-50 候選。
   - 若提供 `--subquery_path`，則用 `Qwen/Qwen3.5-4B` 根據 `query + sub-queries + document` 偵測 over-ranking 風險。
   - 將風險轉成 penalty，修正 reranker 分數。
   - 取 final top-30 當作最終結果。

這裡刻意不做 query expansion，避免 expansion 品質波動影響原始 query 品質。

## 文件表示

這個方法的 document 文字表示不是只看 `Question`，而是：

- `Question + Answer`

也就是說：

- BM25 會在 `Question + Answer` 上做關鍵字匹配
- Dense embeddings 也會用 `Question + Answer`
- Reranker 看到的 document 也是 `Question + Answer`
- BM25 中文斷詞固定使用 `jieba`

query 端則維持使用原始 `Question`。

## 安裝

最小安裝清單已整理在：

- `hybrid_BM25_Dense_Rerank/requirements_hybrid.txt`

可直接安裝：

```bash
pip install -r hybrid_BM25_Dense_Rerank/requirements_hybrid.txt
```

注意：

- 這份檔案預設只放最小必要套件。
- `jieba` 已列為必要套件，BM25 中文斷詞固定使用 `jieba`。
- 如果你要用預設的 Qwen 4-bit GPU 載入方式，請另外安裝 `accelerate` 與 `bitsandbytes`。

## 入口

- `hybrid_BM25_Dense_Rerank/scripts/run_hybrid_union_rerank.py`

## Debug run

```bash
python Hybrid_BM25_Dense/scripts/run_hybrid_union_rerank.py \
  --data data/IR_data.json \
  --query query/phase2_query.json \
  --subquery_path query/phase2_subqueries.json \
  --output outputs/hybrid_union_debug_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --RRF \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --risk_top_k 50 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --risk_model Qwen/Qwen3.5-4B \
  --log_file outputs/logs/hybrid_union_debug.log \
  --cuda_visible_devices 1
```

## Full run

```bash
# union
python Hybrid_BM25_Dense/scripts/run_hybrid_union_rerank.py \
  --data data/IR_data.json \
  --query query/phase2_query.json \
  --subquery_path query/phase2_sub_query.json \
  --output outputs/hybrid_union_query2_LLMrisk_results.json \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --risk_top_k 50 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --risk_model Qwen/Qwen3.5-4B \
  --log_file outputs/logs/hybrid_union_LLMrisk_query2.log \
  --cuda_visible_devices 0 \
  --risk_lambda 0.2

# RRF 的執行指令
python Hybrid_BM25_Dense/scripts/run_hybrid_union_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --subquery_path query/phase1_subqueries.json \
  --output outputs/hybrid_rrf_query1_results.json \
  --RRF \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --risk_top_k 50 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --risk_model Qwen/Qwen3.5-4B \
  --log_file outputs/logs/hybrid_rrf_query1.log \
  --cuda_visible_devices 0

```

## Sub-query 格式

`--subquery_path` 預期是以下 JSON 格式：

```json
[
  {
    "query_id": "Q1",
    "sub_queries": [
      "子查詢 1",
      "子查詢 2",
      "子查詢 3"
    ]
  }
]
```

## Risk penalty

當提供 `--subquery_path` 時，流程會啟用 risk detector：

1. 先對 union / RRF 候選做 reranker，保留 `top-50`。
2. 再對這 50 篇文件，用 `Qwen/Qwen3.5-4B` 評估四種風險：
   - `topic_mismatch_risk`
   - `missing_subquery_risk`
   - `keyword_only_risk`
   - `shallow_answer_risk`
3. 風險會加權成：

```text
risk =
0.35 * missing_subquery_risk
+ 0.30 * topic_mismatch_risk
+ 0.20 * keyword_only_risk
+ 0.15 * shallow_answer_risk
```

4. 最終分數為：

```text
final_score = rerank_score_norm - lambda * risk
```

其中 `lambda` 由 `--risk_lambda` 控制，預設為 `0.1`。

## 重要參數

- `--bm25_top_k`：BM25 第一次檢索保留幾篇文件。
- `--dense_top_k`：Dense 第一次檢索保留幾篇文件。
- `--risk_top_k`：reranker 先保留幾篇文件再做 risk penalty，預設 `50`。
- `--final_top_k`：risk penalty 後最後輸出幾筆。
- `--RRF`：啟用 RRF 融合；未指定時維持原本的 union seed 排序。
- `--rrf_k`：RRF 常數，預設為 `60`。
- `--subquery_path`：sub-query JSON 路徑；提供後會啟用 risk penalty。
- `--risk_model`：risk detector 模型，預設 `Qwen/Qwen3.5-4B`。
- `--risk_model_cache_dir`：risk detector 模型快取根目錄，預設 `/workplace/Share/LLM_model`。
- `--risk_lambda`：風險懲罰權重，預設 `0.1`。
- `--risk_batch_size`：risk detector 批次大小。
- `--risk_max_length` / `--risk_max_new_tokens`：risk detector prompt 長度與生成長度。
- `--bm25_k1` / `--bm25_b`：BM25 參數。
- `--retrieval_instruction`：注入到 dense query 端與 reranker prompt 的 instruction。
- `--disable_4bit`：關閉 4-bit 量化。
- `--no_cache`：不重用 BM25 / dense cache。

## Cache

- BM25 index cache：`outputs/cache/bm25/`
- Dense embeddings cache：`outputs/cache/dense/`
- Model snapshot cache：`outputs/cache/models/`

若使用 `--limit_docs` 做 debug run，腳本結束時會自動刪除本次 debug 的 BM25 與 dense cache，避免後續讀到縮小資料池的快取。

## Output fields

每個 query 的輸出會包含：

- `query_id`
- `query`
- `sub_queries`
- `sub_query_count`
- `fusion_method`
- `rrf_k`
- `bm25_top_k`
- `dense_top_k`
- `fusion_size`
- `union_size`
- `rerank_top_k`
- `final_top_k`
- `risk_penalty_enabled`
- `risk_model_name_or_path`
- `risk_lambda`
- `risk_top_k`
- `document_text_mode`
- `results`

每筆結果會包含：

- `rank`
- `doc_index`
- `doc_id`
- `question`
- `answer`
- `fusion_method`
- `fusion_rank`
- `fusion_score`
- `candidate_sources`
- `union_rank`
- `union_seed_rank`
- `bm25_score`
- `bm25_rank`
- `dense_score`
- `dense_rank`
- `rrf_score`
- `rerank_score_norm`
- `rerank_score`
- `rerank_rank`
- `risk_lambda`
- `risk`
- `risk_parse_ok`
- `topic_mismatch_risk`
- `missing_subquery_risk`
- `keyword_only_risk`
- `shallow_answer_risk`
- `final_score`
- `final_rank`

註記：

- 為了相容既有輸出格式，`union_size` / `union_rank` / `union_seed_rank` 仍會保留。
- 啟用 `--RRF` 時，實際融合排序以 `fusion_method="rrf"` 與 `rrf_score` / `fusion_rank` 為準。
- 若未提供 `--subquery_path`，則不會啟用 risk penalty；此時 `risk=0`，`final_score` 會等於 `rerank_score_norm`。
