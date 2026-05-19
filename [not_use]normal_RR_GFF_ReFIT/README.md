# GFF + ReFIT

這個資料夾目前只保留 `GFF + ReFIT` 流程，對應論文
`Can Query Expansion Improve Generalization of Strong Cross-Encoder Rankers?`
中的 `Generate, Filter, and Fuse (GFF)`，再把 fused feedback 接到 ReFIT。

目前程式的實際流程是：

1. 讀取 document corpus 與 query。
2. 載入 dense retriever，建立或讀取 document embeddings。
3. 對每個 query 做第一次 dense retrieval，取得 feedback top-K candidates。
4. 保留第一次 retrieval 的候選文件索引與 document embeddings，釋放 dense 模型權重。
5. 載入 generator，執行 `Q2D2K`：
   - `Q2D`：先生成多段回答 passage
   - `D2K`：再從每段 passage 抽取 keywords
6. 在每個 passage 抽完 keywords 後，會先做一次 passage-level 過濾，把和原 query 完全相同的詞先排除。
7. 累積多輪生成結果，統計 `keyword_frequencies`。
8. 用 `frequency` 選 final keywords：
   - 優先排除和原 query 完全相同的 keyword
   - 不再把這些重複詞補回，寧可 selected keywords 變少
9. 每個 keyword 個別接回原 query，形成 expanded queries。
10. 載入 reranker，對 original query 與 expanded queries 在同一批 candidates 上打分。
11. 用 reciprocal-rank fusion 融合 expanded-query rankings 與 original-query ranking。
12. 用 fused feedback score 執行 ReFIT，只更新 query embedding。
13. 用更新後的 query embedding 做第二次 dense retrieval，直接輸出 top-N。

也就是說，目前版本：

- `selected_keywords` 是由 `keyword_frequencies` 決定，不再使用 `query-keyword similarity`
- `dense` 模型權重會在第一次 dense retrieval 後先釋放，不留到 keyword selection
- 第二次 retrieval 不需要重新載入 dense 模型，只需要現成的 document embeddings 與更新後的 query embedding

## 預設模型

- Dense retriever: `Qwen/Qwen3-Embedding-4B`
- Reranker: `Qwen/Qwen3-Reranker-4B`
- Generator: `Qwen/Qwen3.5-4B`
- 載入方式：預設使用 bitsandbytes 4-bit (`nf4`) 量化，`compute_dtype=float16`

## Colab 環境建置

你目前在 Colab 使用的安裝方式可直接寫成：

```bash
pip install -U pip
pip install datasets peft sentence-transformers FlagEmbedding rank-bm25 ir_datasets ijson jieba

#對於CUDA版本=12.4的機台
python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

python -m pip install --no-cache-dir --upgrade bitsandbytes accelerate transformers sentencepiece

pip install --upgrade transformers
```

建議補充注意事項：

- `transformers` 需至少支援目前使用的 Qwen tokenizer / chat template 行為。
- 4-bit 量化預設需要 CUDA。
- `--dense_model` / `--reranker_model` / `--generator_model` 可以傳 Hugging Face model id，也可以傳本地模型路徑。
- Qwen 官方建議 query 端帶 instruction；目前腳本預設會加上：
  `給定一個問題，請檢索出語意最相關、問題表述最相近的問題。`
- 模型權重預設會下載到 `outputs/cache/models/`，之後直接從本地快取 snapshot 載入。
- dense document embeddings 會寫入 `outputs/cache/dense/`。

## 入口

- `normal_RR_GFF_ReFIT/scripts/run_gff_refit_retrieval.py`

## Debug run

```bash
python normal_RR_GFF_ReFIT/scripts/run_gff_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase2_query.json \
  --output outputs/qwen_debug_gff_refit_results.json \
  --limit_docs 250 \
  --limit_queries 1 \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --generator_model Qwen/Qwen3.5-4B \
  --gff_rounds 3 \
  --gff_passages_per_round 2 \
  --gff_keywords_per_passage 15 \
  --gff_top_keywords 3 \
  --log_file outputs/logs/qwen_debug_gff_refit.log \
  --cuda_visible_devices 0 \
  --print_query_vectors
```

Debug run 說明：

- `--limit_docs` / `--limit_queries` 用來縮小資料規模，方便觀察 Q2D2K、keyword selection、GFF fusion 與 ReFIT 行為。
- 腳本在 debug run 結束後，會自動刪除本次 dense cache：
  - `.npy`
  - `.json`
  - `.partial.npy`
  - `.partial.json`
- 這是為了避免後續 full run 誤讀到 debug 資料池快取。
- 清理邏輯寫在 `run_gff_refit_retrieval.py::cleanup_debug_dense_cache`。

## Full run

```bash
python normal_RR_GFF_ReFIT/scripts/run_gff_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase2_query.json \
  --output outputs/qwen_gff_refit_results.json \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --generator_model Qwen/Qwen3.5-4B \
  --gff_rounds 3 \
  --gff_passages_per_round 2 \
  --gff_keywords_per_passage 15 \
  --gff_top_keywords 3 \
  --log_file outputs/logs/qwen_gff_refit.log \
  --cuda_visible_devices 0 \
  --print_query_vectors
```

## 重要參數

- `--feedback_top_k`：第一次 dense retrieval 要保留多少 candidates，供 GFF 與 ReFIT feedback 使用。
- `--final_top_k`：第二次 dense retrieval 最後輸出幾筆結果。
- `--refit_updates`：ReFIT 更新 query embedding 的 gradient descent 次數。
- `--refit_learning_rate`：ReFIT 更新 query embedding 的 learning rate。
- `--refit_temperature`：teacher score softmax 使用的 temperature。
- `--generator_model`：Q2D2K 使用的 generator 模型。
- `--gff_rounds`：self-consistency 的輪數。
- `--gff_passages_per_round`：每輪生成幾段 passages。
- `--gff_keywords_per_passage`：每段 passage 最多抽幾個 keywords。
- `--gff_top_keywords`：最後保留幾個 keywords。
- `--gff_rrf_k`：reciprocal-rank fusion 的 `k`。
- `--gff_original_query_weight`：original query ranking 的固定融合權重。
- `--model_cache_dir`：指定模型快取資料夾。
- `--disable_4bit`：關閉 4-bit 量化。
- `--retrieval_instruction`：自訂 query 端與 reranker prompt 使用的 instruction。
- `--print_query_vectors` / `--query_vector_preview_dims`：輸出 query embedding 更新前後摘要。

## Keyword selection 規則

目前 `selected_keywords` 的規則如下：

1. 先做 `Q2D2K` 多輪生成。
2. 每個 passage 抽完 keywords 後，先排除和原 query 完全相同的詞，再累積 candidate keywords。
3. 依 `keyword_frequencies` 由高到低排序。
4. 最後選 `selected_keywords` 時，再次排除和原 query 完全相同的 keyword。
5. 不再把這些重複詞補回，最後的 `selected_keywords` 可能少於 `gff_top_keywords`。

因此：

- 目前不再使用 `keyword_similarity_scores`
- 近義詞、相近詞、拆分詞仍可能保留
- 但和 query 完全相同的字串會被直接排除，不再補回

## 顯存釋放策略

目前腳本刻意把三個模型分段載入與釋放：

1. Dense retriever
   - 建 document embeddings
   - 做第一次 dense retrieval
   - 釋放 dense 模型權重
2. Generator
   - 做 Q2D2K
   - 釋放 generator 模型權重
3. Reranker
   - 做 GFF rerank + fusion
   - 釋放 reranker 模型權重

注意：

- `dense.release_resources()` 釋放的是 dense 模型權重與 tokenizer，不會刪掉 `document embeddings`
- ReFIT 與第二次 dense retrieval 仍可直接使用既有 embeddings，不需要重新載入 dense 模型

## Output fields

每個 query 會輸出以下主要欄位：

- `query_id`：query 的 ID。
- `query`：實際拿去檢索的問題文字。
- `feedback_top_k`：第一次 dense retrieval 取出的候選數。
- `final_top_k`：最後輸出的第二次 dense retrieval 結果數。
- `refit_updates`：ReFIT 更新 query embedding 的次數。
- `refit_learning_rate`：ReFIT 的 learning rate。
- `refit_temperature`：teacher score softmax 的 temperature。
- `refit_use_minmax`：是否先做 min-max normalization。
- `query_vector_shift_l2`：ReFIT 前後 query embedding 的 L2 距離。
- `dense_model_name_or_path`：本次使用的 dense model。
- `reranker_model_name_or_path`：本次使用的 reranker model。
- `generator_model_name_or_path`：本次使用的 generator model。
- `use_4bit`：本次是否用 4-bit 量化載入。
- `compute_dtype`：本次推論使用的 compute dtype。
- `retrieval_instruction`：注入到 Qwen query 端與 reranker prompt 的 instruction。
- `gff.query_expansion`：Q2D2K passages、candidate keywords、selected keywords、keyword frequencies。
- `gff.first_stage_fusion`：第一次 dense candidates 的 fusion 摘要。
- `query_embedding_debug`：開啟 `--print_query_vectors` 時才會輸出。
- `refit_results`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的 top-N。

單筆結果內常見欄位：

- `rank`：`refit_results` 中的排名。
- `doc_index`：文件在載入後 document list 中的 index。
- `doc_id`：原始資料中的文件 ID。
- `question` / `answer`：被檢索到的歷史問答內容。
- `first_dense_score`：第一次 dense retrieval 的內積分數。
- `first_dense_rank`：第一次 dense retrieval 的排名。
- `first_original_rerank_score`：original query 在第一次 rerank 的分數。
- `first_original_rerank_rank`：original query 在第一次 rerank 的排名。
- `first_gff_fused_score`：第一次 GFF fusion 分數。
- `first_gff_fused_rank`：第一次 GFF fusion 排名。
- `refit_dense_score`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的內積分數。
- `refit_dense_rank`：ReFIT 更新 query embedding 後，第二次 dense retrieval 的排名。

## Query embedding debug

若要查看更新前和 ReFIT 更新後的 query embedding 摘要，可以加：

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
