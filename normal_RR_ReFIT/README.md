# Dense-only ReFIT

這個資料夾實作 ReFIT 的核心流程，不混合 BM25：

1. 用 dense retriever 對 query 做第一次檢索，取得 feedback top-K。
2. 用 reranker 對同一批 dense candidates 計算 relevance feedback 分數。
3. 將 reranker 分數做 min-max normalization，再以 temperature `T=2` 轉成 softmax 分布。
4. 將 retriever 內積分數做 min-max normalization，再轉成 softmax 分布。
5. 用 KL divergence 更新 query embedding，預設 `n=100`、learning rate `0.005`。
6. 用更新後的 query embedding 做第二次 dense retrieval，第二次結果不再 rerank。

目前預設模型：

- Dense retriever: `Qwen/Qwen3-Embedding-4B`
- Reranker: `Qwen/Qwen3-Reranker-4B`
- 載入方式：預設使用 bitsandbytes 4-bit (`nf4`) 量化，`compute_dtype=float16`

注意事項：

- `transformers` 需至少 `4.51.0` 才支援 `qwen3`；目前環境若低於此版本會出現 `KeyError: 'qwen3'`。
- 4-bit 量化預設需要 CUDA。
- `--dense_model` / `--reranker_model` 可以傳 Hugging Face model id，也可以傳本地模型路徑。
- Qwen 官方建議 query 端帶 instruction；目前腳本預設會加上 `給定一個使用者提出的問題，請檢索語意最相關、問題表述最相近的歷史問題。只根據 Question 欄位判斷相關性，不要考慮 Answer 內容。`。
- dense document embeddings 會寫入 `outputs/cache/dense/`；若中途停止，重新用相同設定執行時會自動從 partial cache 繼續，不需要整個資料庫重編碼。

## Debug run

```bash
python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_debug_refit_1000docs_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/qwen_debug_refit_1000docs.log \
  --cuda_visible_devices 1 \
  --print_query_vectors \
  --query_vector_preview_dims 10
```

## Full run

```bash
python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_refit_query1_results.json \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/qwen_refit_query1.log \
  --cuda_visible_devices 1 \
  --print_query_vectors \
  --query_vector_preview_dims 10
```

若要指定 GPU，可以加上：

```bash
--cuda_visible_devices 3
```

若要暫時關掉 4-bit 量化，可以加上：

```bash
--disable_4bit
```

若要改成別的 Qwen 指令文字，可以加上：

```bash
--retrieval_instruction "給定一個使用者提出的問題，請檢索語意最相關、問題表述最相近的歷史問題。只根據 Question 欄位判斷相關性，不要考慮 Answer 內容。"
```

## Output fields

每個 query 會輸出以下主要欄位：

- `query_id`：query 的 ID。
- `query`：實際拿去檢索的問題文字。
- `feedback_top_k`：第一次 dense retrieval 取出、交給 reranker feedback 的候選數。
- `final_top_k`：最後輸出的 ReFIT 第二次 dense retrieval 結果數。
- `refit_updates`：ReFIT 更新 query embedding 的 gradient descent 次數。
- `refit_learning_rate`：只更新 query embedding 時使用的 learning rate。
- `refit_temperature`：reranker 分數 softmax 使用的 temperature，預設是 `2.0`。
- `refit_use_minmax`：是否先對 reranker/retriever 分數做 min-max normalization。
- `query_vector_shift_l2`：ReFIT 前後 query embedding 的 L2 距離，數值越大代表 query 向量改動越多。
- `dense_model_name_or_path`：本次使用的 dense model。
- `reranker_model_name_or_path`：本次使用的 reranker model。
- `use_4bit`：本次是否用 4-bit 量化載入。
- `compute_dtype`：本次推論使用的 compute dtype。
- `retrieval_instruction`：注入到 Qwen query 端與 reranker prompt 的指令文字。
- `query_embedding_debug`：開啟 `--print_query_vectors` 時才會輸出，記錄 query embedding 更新前後的向量摘要。
- `refit_results`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的 top-N。

單筆結果內常見欄位：

- `rank`：`refit_results` 中的排名，代表 ReFIT 後第二次 dense retrieval 的排名。
- `doc_index`：文件在載入後 document list 中的 index。
- `doc_id`：原始資料中的文件 ID。
- `question` / `answer`：被檢索到的歷史問答內容。
- `first_dense_score`：第一次 dense retrieval 的內積分數。
- `first_dense_rank`：第一次 dense retrieval 的排名。
- `rerank_feedback_score`：第一次 reranker 對 query-document pair 的 relevance feedback 分數。
- `first_rerank_rank`：第一次 dense candidates 經 reranker 排序後的排名。
- `refit_dense_score`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的內積分數。
- `refit_rank`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的排名；數值會和該清單的 `rank` 相同。

## Query embedding debug

query 更新前後的值是 dense retriever 產生的 query embedding，也就是一個高維向量。若要查看更新前和 ReFIT 更新後的數值，可以加：

```bash
--print_query_vectors \
--query_vector_preview_dims 10
```

這會在 log 和 JSON 的 `query_embedding_debug` 中輸出：

- `embedding_dim`：完整 query embedding 的維度。
- `preview_dims`：本次實際印出的前幾個維度。
- `original_norm_l2`：更新前 query embedding 的 L2 norm。
- `updated_norm_l2`：ReFIT 更新後 query embedding 的 L2 norm。
- `shift_l2`：更新前後兩個向量的 L2 距離。
- `original_query_vector_preview`：更新前 query embedding 的前 N 維。
- `updated_query_vector_preview`：更新後 query embedding 的前 N 維。
- `delta_vector_preview`：更新後減去更新前的前 N 維差值。

若真的想印完整向量，可以把 `--query_vector_preview_dims` 設成 embedding 維度，但 log 和 JSON 會變大。
