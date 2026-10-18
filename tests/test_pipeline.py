import numpy as np

from legal_assistant import pipeline_core as core


def hit(index, article="Điều 1"):
    return {"doc_index": index, "doc_id": str(index), "score": 1,
            "doc": {"doc_id": str(index), "article_ref": article, "article_title": article,
                    "doc_code": "04/2017/QH14", "doc_title": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                    "canonical_doc_key": "04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
                    "content": "Một nội dung thử nghiệm.", "bm25_search": "doanh nghiệp nhỏ"}}


def test_rrf_combines_ranks_without_duplicating_articles(monkeypatch):
    hits = [hit(0), hit(1), hit(2)]
    monkeypatch.setattr(core, "docs", [row["doc"] for row in hits])
    result = core.rrf_fusion([[hits[0], hits[1]], [hits[1], hits[2]]], k=60)
    assert result[0]["doc_index"] == 1
    assert len(result) == 3
    assert np.isclose(result[0]["rrf_score"], 1/62 + 1/61)


def test_bm25_small_corpus_does_not_overflow_top_k():
    class Index:
        def get_scores(self, tokens):
            return [0.1, 2.0]
    result = core.bm25_search("doanh nghiệp", Index(), [hit(0)["doc"], hit(1)["doc"]], top_k=200)
    assert [row["doc_index"] for row in result] == [1, 0]


def test_vietnamese_normalization():
    assert core.strip_diacritics("Đăng ký doanh nghiệp") == "Dang ky doanh nghiep"
    assert core.bm25_tokenize("Doanh nghiệp nhỏ")


def test_format_uses_document_and_article_identity():
    output = core.format_competition_output("Câu hỏi?", "Câu trả lời.", [hit(0)], qid=7)
    assert output["id"] == 7
    assert output["relevant_articles"] == ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 1"]


def test_cleanup_removes_internal_labels():
    result = core.normalize_answer_lead("Hồ sơ gồm gì?", "Có, [Tài liệu 1] hồ sơ gồm đơn đề nghị.")
    assert "Tài liệu 1" not in result


def test_reranker_uses_real_candidate_order_and_scores(monkeypatch):
    class Reranker:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 3
            return [0.1, 0.9, 0.5]
    monkeypatch.setattr(core, "reranker", Reranker(), raising=False)
    result = core.rerank_hits("question", [hit(0), hit(1), hit(2)], top_k=2)
    assert [row["doc_index"] for row in result] == [1, 2]


def test_long_first_article_is_not_silently_dropped():
    candidate = hit(0)
    candidate["doc"]["content"] = "Nội dung " * 3000
    llm = core.LLMInterface(max_input_chars=1000)
    prompt = llm._build_prompt("Câu hỏi?", [candidate])
    assert "Nội dung Nội dung" in prompt
    assert "Câu hỏi?" in prompt


def test_hybrid_family_and_dense_rankings_share_article_identity(monkeypatch):
    candidates = [hit(0), hit(1), hit(2)]
    monkeypatch.setattr(core, "docs", [x["doc"] for x in candidates])
    class Index:
        def get_scores(self, tokens):
            return [0.2, 0.9, 0.1]
    class Encoder:
        def encode(self, *args, **kwargs):
            return np.array([[1.0, 0.0]], dtype="float32")
    monkeypatch.setattr(core, "bm25_normal", Index(), raising=False)
    monkeypatch.setattr(core, "bm25_no_diac", Index(), raising=False)
    monkeypatch.setattr(core, "dense_model", Encoder(), raising=False)
    monkeypatch.setattr(core, "dense_embeddings", np.array([[0.1, 0], [0.2, 0], [1, 0]]), raising=False)
    result = core.hybrid_dense_bm25_search("doanh nghiệp", bm25_top_k=200, dense_top_k=200)
    assert result["bm25_family_hits"][0]["doc_index"] == 1
    assert result["dense_hits"][0]["doc_index"] == 2
    assert {x["doc_index"] for x in result["final_fused_hits"]} == {0, 1, 2}


def test_citation_selection_only_returns_retrieved_sources(monkeypatch):
    monkeypatch.setitem(core.CONFIG, "citation_verifier_enabled", False)
    candidates = [hit(0, "Điều 4"), hit(1, "Điều 5")]
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = 0.99 - i * 0.01
    selected = core.select_output_hits("Theo Điều 4 Luật Hỗ trợ doanh nghiệp nhỏ và vừa.",
                                       candidates, question="Doanh nghiệp nhỏ và vừa là gì?")
    assert selected and all(item in candidates for item in selected)
    assert len(selected) <= 3
