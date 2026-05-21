# Hybrid BM25 + Dense Union Rerank

這個資料夾提供一條和 `normal_RR_GFF_ReFIT` 平行、彼此獨立的新方法線，不會修改原本的 `GFF + ReFIT` 程式。

此方法的核心流程如下：

1. 讀取 `data/IR_data.json` 與 query JSON。
2. 將每篇 document 組成 `Question + Answer` 文字。
3. 建立或讀取 BM25 index。
4. 建立或讀取 dense document embeddings。
5. 對每個 query：
   - 用 BM25 取 top-100 關鍵字候選。
   - 用 dense retrieval 取 top-100 語意候選。
   - 將兩邊結果做 union，得到一個可能大於 100 的候選集合。
   - 用 reranker 對 union candidates 重新排序。
   - 取 rerank top-30 當作最終結果。

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
  --output outputs/hybrid_union_debug_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/hybrid_union_debug.log \
  --cuda_visible_devices 1
```

## Full run

```bash
python Hybrid_BM25_Dense/scripts/run_hybrid_union_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/hybrid_union_query1_results.json \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/hybrid_union_query1.log \
  --cuda_visible_devices 0
```

## 重要參數

- `--bm25_top_k`：BM25 第一次檢索保留幾篇文件。
- `--dense_top_k`：Dense 第一次檢索保留幾篇文件。
- `--final_top_k`：rerank 後最後輸出幾筆。
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
- `bm25_top_k`
- `dense_top_k`
- `union_size`
- `final_top_k`
- `document_text_mode`
- `results`

每筆結果會包含：

- `rank`
- `doc_index`
- `doc_id`
- `question`
- `answer`
- `candidate_sources`
- `union_rank`
- `union_seed_rank`
- `bm25_score`
- `bm25_rank`
- `dense_score`
- `dense_rank`
- `rerank_score`
- `rerank_rank`
