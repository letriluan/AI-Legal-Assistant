"""Load trusted local artifacts only when full RAG is explicitly requested."""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import ollama

from .settings import CONFIG_DIR, data_dir


class ArtifactError(ValueError):
    pass


def verify_artifacts(directory, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for name in ("bm25_dual_with_docs_phapdien64_web_canonical.pkl", "dense_bge_m3_phapdien64_web.npy"):
        path = directory / name
        if not path.is_file():
            raise ArtifactError("Required local artifact missing: " + name)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest[name]["sha256"]:
            raise ArtifactError("Artifact checksum mismatch: " + name)
    return manifest


class RagEngine:
    def __init__(self):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        from . import pipeline_core as core

        directory = data_dir()
        verify_artifacts(directory, os.getenv("ARTIFACT_MANIFEST", str(CONFIG_DIR / "artifact_manifest.json")))
        config = json.loads((CONFIG_DIR / "submission_v6_3.json").read_text(encoding="utf-8"))
        core.CONFIG.clear()
        core.CONFIG.update(config)
        # Pickle can execute code. Only the exact user-supplied, checksummed bundle is accepted.
        with (directory / config["bm25_bundle_path"]).open("rb") as handle:
            bundle = pickle.load(handle)
        core.docs = bundle["docs"]
        core.doc_ids = bundle["doc_ids"]
        if len(core.docs) != len(core.doc_ids) or any(
            doc.get("doc_id") != identity for doc, identity in zip(core.docs, core.doc_ids)):
            raise ArtifactError("BM25 document identities are not aligned")
        core.bm25_normal = bundle["bm25_normal"]
        core.bm25_no_diac = bundle["bm25_no_diac"]
        core.dense_embeddings = np.load(directory / config["dense_emb_path"], mmap_mode="r", allow_pickle=False)
        if core.dense_embeddings.ndim != 2 or core.dense_embeddings.shape != (len(core.docs), 1024):
            raise ArtifactError("Embedding dimensions do not match corpus rows / BGE-M3")
        local = os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() == "true"
        device = os.getenv("DEVICE", "cpu")
        core.dense_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                                               device=device, local_files_only=local)
        core.dense_model.max_seq_length = 512
        core.reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                                    device=device, max_length=512, local_files_only=local)
        core.DOC_NAME_MAP = core.build_doc_name_map(core.docs)
        core.ollama = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                                    timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")))
        core.llm = core.LLMInterface(model_name=config["llm_model"],
                                    max_input_chars=config["llm_max_input_chars"],
                                    max_new_tokens=config["llm_max_new_tokens"], temperature=0)
        self.core = core
        self.config = config

    def ask(self, question):
        core, cfg = self.core, self.config
        started = time.perf_counter()
        retrieved = core.hybrid_dense_bm25_search(question, **{
            name: cfg[name] for name in ("bm25_top_k", "dense_top_k", "bm25_family_top_k",
                                         "final_fused_top_k", "rrf_k")})
        kept = core.rerank_hits(question, retrieved["final_fused_hits"],
                               top_k=cfg["final_top_k"], rerank_input_k=cfg["rerank_input_k"],
                               batch_size=cfg["rerank_batch_size"])
        answer = core.llm.generate(question, kept)
        if cfg.get("strip_extra_dieu"):
            answer = core.strip_uncited_dieu(answer, kept)
        answer = core.normalize_answer_lead(question, answer)
        selected = core.select_output_hits(answer, kept, max_articles=cfg["max_output_articles"],
                                           fallback_articles=cfg["fallback_output_articles"], question=question)
        result = core.format_competition_output(question, answer, selected, qid=None)
        sources = []
        for hit in selected:
            doc = hit["doc"]
            code, title = core.extract_doc_code_name(doc)
            sources.append({"document": title, "code": code, "article": core.extract_article_ref(doc),
                            "excerpt": (doc.get("content") or "")[:1600]})
        return {"question": question, "answer": result["answer"], "sources": sources,
                "mode": "rag", "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
