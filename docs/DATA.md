# Data and model setup

The original archive remains unchanged. The local preparation points its ignored `.env`
to the extracted `01_data/data_for_run_full_final_pipeline` directory. No large files
are copied into the prepared Git repository.

| Artifact | Size (bytes) | Role |
| --- | ---: | --- |
| articles_phapdien64_web_canonical.jsonl | 517,717,165 | Canonical article records |
| bm25_dual_with_docs_phapdien64_web_canonical.pkl | 703,010,716 | Two BM25 indexes and ordered documents |
| dense_bge_m3_phapdien64_web.npy | 291,020,928 | BGE-M3 dense vectors |
| R2AIStage1DATA.json | 530,758 | Original batch questions |

`runtime.verify_artifacts` checks the BM25 and embedding SHA-256 hashes against the
bundled manifest before loading them. Pickle is executable serialization: a checksum
is useful only when the manifest is trusted. Do not replace the manifest with one
provided alongside an untrusted pickle and assume it is safe. The runtime verifies
BM25 document identities and embedding dimensions; the local preparation separately
verified the entire JSONL order against the BM25 bundle.

Models:

- `BAAI/bge-m3` via SentenceTransformer (max sequence length 512)
- `BAAI/bge-reranker-v2-m3` via CrossEncoder (max length 512)
- `qwen2.5:7b` via Ollama, for generation and selective citation verification

The original full corpus is a mixture of legal records and collected web documents.
There is no complete redistribution licence or official scoring definition in the
provided package. Keep the full corpus and model artifacts local until the relevant
permissions and competition terms are confirmed. Source documentation is retained
in SUBMISSION_DATA.md and SUBMISSION_MODELS.md; references to submission folders in
those files describe the original package, not this repository layout.
