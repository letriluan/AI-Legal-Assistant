import numpy as np
"""Original retrieval and citation algorithms extracted from the submitted notebook.

Model/data initialization and the automatic 2,000-question run are removed.
See docs/PROVENANCE.md for extraction details and limits.
"""

CONFIG = {'exp_name': 'canonical_v6_3_same_doc_band90_92', 'articles_path': 'articles_phapdien64_web_canonical.jsonl', 'bm25_bundle_path': 'bm25_dual_with_docs_phapdien64_web_canonical.pkl', 'dense_emb_path': 'dense_bge_m3_phapdien64_web.npy', 'test_path': 'R2AIStage1DATA.json', 'bm25_top_k': 200, 'dense_top_k': 200, 'bm25_family_top_k': 160, 'final_fused_top_k': 200, 'rrf_k': 60, 'use_reranker': True, 'rerank_input_k': 160, 'final_top_k': 8, 'rerank_batch_size': 32, 'llm_model': 'qwen2.5:7b', 'llm_temperature': 0.0, 'llm_max_new_tokens': 420, 'llm_max_input_chars': 8000, 'strip_extra_dieu': True, 'inject_missing_citations': False, 'max_output_articles': 3, 'fallback_output_articles': 1, 'postprocess_selector': 'v6_3_same_doc_band90_92_over_v5', 'citation_verifier_enabled': True, 'citation_verifier_model': 'qwen2.5:7b', 'citation_verifier_num_predict': 6, 'citation_verifier_max_content_chars': 1200, 'citation_verifier_veto_only': True, 'citation_verifier_drop_support_on_single': False, 'v63_band_low': 0.9, 'v63_band_high': 0.92, 'v63_band_max_rank': 4, 'v63_same_doc_only': True}

docs = []
DOC_NAME_MAP = {}

# Extracted from notebook cell 3
import re

import unicodedata

from rank_bm25 import BM25Okapi

from pyvi import ViTokenizer

def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for robust matching."""
    if text is None:
        return ""

    #text = str(text)
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", text)

    return "".join(
        ch for ch in nfkd
        if unicodedata.category(ch) != "Mn"
    )

def normalize_text(text):
    if text is None:
        return ""

    #text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def bm25_tokenize(text, remove_diacritics=False):
    text = normalize_text(text)

    if remove_diacritics:
        text = strip_diacritics(text)

    text = ViTokenizer.tokenize(text)
    return text.split()

# Extracted from notebook cell 5
import numpy as np

def bm25_search(query, bm25_index, docs, top_k=10, remove_diacritics=False):
    query_tokens = bm25_tokenize(query, remove_diacritics=remove_diacritics)
    scores = np.asarray(bm25_index.get_scores(query_tokens))

    top_k = min(max(int(top_k), 0), len(scores))
    if not top_k:
        return []
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        {
            "doc_id": docs[int(idx)]["doc_id"],
            "doc_index": int(idx),
            "score": float(scores[int(idx)]),
            "doc": docs[int(idx)],
        }
        for idx in top_indices
    ]

# Extracted from notebook cell 12
def dense_search(query, top_k=100):
    query_emb = dense_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")[0]

    scores = dense_embeddings @ query_emb

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "doc_id": docs[idx]["doc_id"],
            "doc_index": int(idx),
            "score": float(scores[idx]),
            "doc": docs[idx],
        })

    return results

# Extracted from notebook cell 13
def rrf_fusion(result_lists, k=60):
    """
    result_lists:
    [
        [ {"doc_index": ..., "doc": ..., "score": ...}, ... ],
        ...
    ]

    Returns:
    [
        {"doc_index": ..., "doc": ..., "rrf_score": ...},
        ...
    ]
    """
    scores = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            doc_index = hit["doc_index"]

            if doc_index not in scores:
                scores[doc_index] = {
                    "doc_index": doc_index,
                    "doc_id": hit.get("doc_id", docs[doc_index].get("doc_id")),
                    "doc": hit.get("doc", docs[doc_index]),
                    "rrf_score": 0.0,
                }

            scores[doc_index]["rrf_score"] += 1.0 / (k + rank)

    fused = list(scores.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused

# Extracted from notebook cell 14
def hybrid_dense_bm25_search(
    query,
    bm25_top_k=100,
    dense_top_k=100,
    bm25_family_top_k=100,
    final_fused_top_k=80,
    rrf_k=60,
):
    # 1. BM25 normal
    bm25_normal_hits = bm25_search(
        query=query,
        bm25_index=bm25_normal,
        docs=docs,
        top_k=bm25_top_k,
        remove_diacritics=False
    )

    # 2. BM25 no-diacritics
    bm25_no_diac_hits = bm25_search(
        query=query,
        bm25_index=bm25_no_diac,
        docs=docs,
        top_k=bm25_top_k,
        remove_diacritics=True
    )

    # 3. Fuse two BM25 retrievers into one BM25 family
    bm25_family_hits = rrf_fusion(
        [
            bm25_normal_hits,
            bm25_no_diac_hits,
        ],
        k=rrf_k
    )[:bm25_family_top_k]

    # 4. Dense retrieval
    dense_hits = dense_search(
        query=query,
        top_k=dense_top_k
    )

    # 5. Fuse BM25 family with dense
    final_fused_hits = rrf_fusion(
        [
            bm25_family_hits,
            dense_hits,
        ],
        k=rrf_k
    )[:final_fused_top_k]

    return {
        "bm25_normal_hits": bm25_normal_hits,
        "bm25_no_diac_hits": bm25_no_diac_hits,
        "bm25_family_hits": bm25_family_hits,
        "dense_hits": dense_hits,
        "final_fused_hits": final_fused_hits,
    }

# Extracted from notebook cell 17
def get_rerank_text(doc, max_chars=2500):
    """
    Text used by reranker.
    Keep it concise because cross-encoder is slower than dense retrieval.
    """
    return doc.get("bm25_search", "")[:max_chars]

def rerank_hits(
    query,
    hits,
    top_k=3,
    rerank_input_k=50,
    batch_size=8,
):
    """
    query: user question
    hits: final_fused_hits from hybrid_dense_bm25_search()
    top_k: final number of articles to keep
    rerank_input_k: how many fused candidates to rerank
    """

    candidates = hits[:rerank_input_k]

    pairs = [
        [query, get_rerank_text(hit["doc"])]
        for hit in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False
    )

    reranked = []

    for hit, score in zip(candidates, scores):
        new_hit = dict(hit)
        new_hit["rerank_score"] = float(score)
        reranked.append(new_hit)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]

# Extracted from notebook cell 19
import ollama

import re

from typing import List, Dict

class LLMInterface:
    SYSTEM_PROMPT = (
        "Bạn là trợ lý pháp lý AI cho doanh nghiệp SME. "
        "Trả lời bằng tiếng Việt có dấu, chính xác, rõ ràng, chỉ dựa trên context pháp luật được cung cấp. "
        "Không suy đoán và không dùng kiến thức ngoài context. "
        "Đọc quy phạm theo nghĩa bắt buộc: nếu điều luật định nghĩa 'X là ...' hoặc quy định 'phải/được/không được', không được đổi thành tùy chọn hoặc đảo nghĩa. "
        "Trích dẫn điều luật tự nhiên theo dạng 'theo Điều <số> <tên văn bản>' khi điều đó trực tiếp hỗ trợ câu trả lời."
    )

    def __init__(
        self,
        model_name="qwen2.5:7b",
        max_input_chars=6000,
        max_new_tokens=160,
        temperature=0.0,
    ):
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _doc_article(self, doc):
        # real article ref so the model cites correct numbers
        # (not phap-dien IDs like "Dieu 1.3.LQ.11")
        try:
            ref = extract_article_ref(doc)
            if ref:
                return ref
        except NameError:
            pass
        return (
            doc.get("article")
            or doc.get("article_number")
            or doc.get("article_title")
            or ""
        )

    def _doc_source(self, doc):
        # proper document name + code so the model cites real document names
        try:
            code, name = extract_doc_code_name(doc)
            if code:
                return f"{name} (số {code})"
        except NameError:
            pass
        return (
            doc.get("source_note")
            or doc.get("doc_name")
            or doc.get("subject_title")
            or doc.get("topic_title")
            or ""
        )

    def _build_prompt(self, query, context_docs):
        parts = []
        total = 0

        for i, hit in enumerate(context_docs, start=1):
            doc = hit["doc"]

            article = self._doc_article(doc)
            title = doc.get("article_title", "")
            source = self._doc_source(doc)
            content = doc.get("content") or doc.get("bm25_search", "")

            s = (
                f"[Tài liệu {i}]\n"
                f"Điều/Ký hiệu: {article}\n"
                f"Tiêu đề điều: {title}\n"
                f"Nguồn: {source}\n"
                f"Nội dung: {content}\n"
            )

            if total + len(s) > self.max_input_chars:
                if not parts:
                    parts.append(s[:self.max_input_chars])
                break

            parts.append(s)
            total += len(s)

        ctx = "\n---\n".join(parts)

        return (
            "Dựa vào các quy định pháp luật sau, hãy trả lời câu hỏi cho doanh nghiệp SME.\n\n"
            f"{ctx}\n\n"
            f"Câu hỏi: {query}\n\n"
            "Yêu cầu bắt buộc:\n"
            "- Viết câu trả lời bằng tiếng Việt có dấu; dùng 'số', 'Có', 'Không', không dùng 'so', 'Co', 'Khong'.\n"
            "- Nêu kết luận trực tiếp ngay câu đầu tiên.\n"
            "- Chỉ bắt đầu bằng 'Có' hoặc 'Không' khi câu hỏi thật sự là câu có/không, ví dụ có cụm 'có ... không', 'có được', 'có phải', 'được phép', 'bắt buộc'. Với câu hỏi hỏi mức, thời hạn, hồ sơ, điều kiện thì không mở đầu bằng 'Có' hoặc 'Không'.\n"
            "- Nếu văn bản định nghĩa chủ thể/hành vi theo dạng 'X là ...' hoặc liệt kê điều kiện cấu thành, hãy xem đó là quy định xác định bắt buộc, không suy thành 'có thể' hay 'không bắt buộc'.\n"
            "- Nếu câu hỏi hỏi mức tiền, tỷ lệ, thời hạn, số ngày, số người, hồ sơ hoặc điều kiện, phải nêu đủ con số/điều kiện chính.\n"
            "- Với câu hỏi về xử phạt/xử lý, chỉ nêu mức phạt hoặc biện pháp xử lý khi context có điều xử phạt rõ ràng; nếu chỉ có điều cấm/nghĩa vụ thì nói rõ hành vi bị cấm hoặc nghĩa vụ bị vi phạm, không tự bịa mức phạt.\n"
            "- Khi có nhiều tài liệu, chọn điều khớp nhất với trọng tâm câu hỏi; không dùng điều chỉ cùng chủ đề nhưng trả lời sang chế độ khác. Nếu có cả quy định hiện hành và quy định cũ cùng vấn đề, ưu tiên quy định hiện hành hoặc quy định khớp với thời điểm câu hỏi.\n"
            "- Trả lời đủ ý trong 2-4 câu; không quá ngắn đến mức thiếu điều kiện hoặc ngoại lệ quan trọng.\n"
            "- Chỉ trích dẫn điều luật thật sự dùng để trả lời. Không thêm danh sách 'Căn cứ pháp lý' nếu các điều đó không trực tiếp cần thiết.\n"
            "- Không nhắc nhãn context nội bộ như 'Tài liệu 1', '[Tài liệu 2]', 'Ký hiệu ...' trong câu trả lời.\n"
            "- Nếu context không đủ để trả lời chắc chắn, nói rõ phần còn thiếu thay vì đoán.\n\n"
            "Trả lời:"
        )

    def generate(self, query, context_docs):
        prompt = self._build_prompt(query, context_docs)

        resp = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "num_ctx": 4096,
            },
        )

        return resp["message"]["content"].strip()

LLMInterface.SYSTEM_PROMPT += (
    " Chỉ mở đầu bằng 'Có' hoặc 'Không' nếu câu hỏi là yes/no thật sự "
    "như 'có ... không', 'có được ... không', 'có bị ... không', "
    "'có phải ... không', 'có cần ... không'. "
    "Nếu câu hỏi hỏi 'làm gì', 'như thế nào', 'bao lâu', 'điều kiện nào', "
    "'nội dung gì', 'hồ sơ gồm gì', hoặc 'mức phạt bao nhiêu', hãy trả lời "
    "trực tiếp và không mở đầu bằng 'Có' hoặc 'Không'."
)

# Extracted from notebook cell 20
import re

import unicodedata

from typing import Dict, List

def clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()

def clean_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip(" ()[];,.-")

def extract_article_ref(doc):
    """Extract the real legal article ref, preferring normalized fields."""
    if doc.get("article_ref"):
        return clean_text(doc.get("article_ref"))

    dieu = "Điều"
    pattern = rf"\b(?:{dieu}|Dieu)\s+(\d+[a-zA-ZđĐ]?)\b"
    for field in ("source_note", "article_title", "bm25_search", "source", "content"):
        m = re.search(pattern, clean_text(doc.get(field, "")), flags=re.IGNORECASE)
        if m:
            return f"{dieu} {m.group(1)}"
    return ""

def _allowed_dieu_numbers(kept):
    allowed = set()
    for hit in kept:
        doc = hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit
        ref = extract_article_ref(doc)
        if ref:
            allowed.add(ref.split()[1].lower())
    return allowed

def strip_uncited_dieu(answer: str, kept: List[Dict]) -> str:
    """
    Remove 'Điều X' mentions NOT belonging to kept docs.
    v2: also matches pháp điển IDs ('Điều 1.3.LQ.11'), eats orphaned
    'khoản N' prefixes and trailing 'của', and tidies leftovers.
    """
    allowed = _allowed_dieu_numbers(kept)

    def repl(m):
        if m.group("num").lower() in allowed:
            # normalize: "Điều 12.3.LQ.12" -> "Điều 12" (keep khoản prefix / "của")
            return f"{m.group('khoan') or ''}Điều {m.group('num')}{m.group('cua') or ''}"
        return ""

    pat = (r"(?P<khoan>[Kk]hoản\s+\d+\s+)?[Đđ]iều\s+(?P<num>\d+[a-zA-Z]?)"
           r"(?P<tail>(?:\.\w+)*)(?P<cua>\s+của)?")
    out = re.sub(pat, repl, answer or "")
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s+([,.;)])", r"\1", out)
    out = re.sub(r"\bTheo\s*,", "Theo quy định,", out)
    out = re.sub(r"\btại\s+và\s+", "tại ", out)
    out = re.sub(r"\bvà\s+và\b", "và", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()

def _article_number(article: str) -> str:
    dieu = "Điều"
    m = re.search(rf"\b(?:{dieu}|Dieu)\s+(\d+[a-zA-ZđĐ]?)\b", article or "", flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""

def _answer_mentions_article(answer: str, article: str) -> bool:
    num = _article_number(article)
    if not num:
        return False
    dieu = "Điều"
    # Match both Vietnamese "Dieu" with accents and ASCII "Dieu" while avoiding 70 or 7.1.
    return re.search(rf"(?<!\d)(?:{dieu}|Dieu)\s+{re.escape(num)}(?![\d.])", answer or "", flags=re.IGNORECASE) is not None

def _has_doc_identity(doc) -> bool:
    code, name = extract_doc_code_name(doc)
    return bool(code and name)

def _is_yes_no_question(question: str) -> bool:
    q = clean_answer(question).lower()
    patterns = [
        r"\bcó\b.+\bkhông\b",
        r"\bcó\s+được\b.+\bkhông\b",
        r"\bcó\s+bị\b.+\bkhông\b",
        r"\bcó\s+phải\b.+\bkhông\b",
        r"\bcó\s+cần\b.+\bkhông\b",
        r"\bđược\s+phép\b.+\bkhông\b",
        r"\bphải\b.+\bkhông\b",
        r"\bcần\b.+\bkhông\b",
    ]
    return any(re.search(p, q) for p in patterns)

def _mentioned_doc_codes(answer: str) -> set:
    codes = set(re.findall(_CODE_RE, answer or "", flags=re.IGNORECASE))
    return {c.upper() for c in codes}

def inject_citations(answer: str, kept: List[Dict], max_missing: int = 1) -> str:
    """
    Conservative optional fallback. Disabled by default in CONFIG.
    Only append the top missing citation when the answer has no explicit Điều mention.
    """
    answer = clean_answer(answer)
    if re.search(r"\bĐiều\s+\d+[a-zA-ZđĐ]?\b", answer, flags=re.IGNORECASE):
        return answer

    missing, seen = [], set()
    for hit in kept:
        doc = hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit
        article = extract_article_ref(doc)
        code, name = extract_doc_code_name(doc)
        if article and code and name:
            item = f"{article} {name}"
            if item not in seen:
                seen.add(item)
                missing.append(item)
        if len(missing) >= max_missing:
            break

    if missing:
        answer = answer.rstrip(".") + ". Căn cứ pháp lý: " + "; ".join(missing) + "."
    return answer

def format_competition_output(question, answer, kept, qid):
    relevant_docs, relevant_articles = [], []
    seen_doc_codes, seen_article_keys = set(), set()

    for hit in kept:
        doc = hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit
        code, name = extract_doc_code_name(doc)
        article = extract_article_ref(doc)
        if not code or not name:
            continue

        # Competition keys should not duplicate the same legal document under two names.
        if code not in seen_doc_codes:
            seen_doc_codes.add(code)
            relevant_docs.append(f"{code}|{name}")

        if article:
            article_key = (code, _article_number(article) or article.lower())
            if article_key not in seen_article_keys:
                seen_article_keys.add(article_key)
                relevant_articles.append(f"{code}|{name}|{article}")

    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "relevant_docs": relevant_docs,
        "relevant_articles": relevant_articles,
    }

def _demo_norm(text: str) -> str:
    text = str(text or "").replace(chr(272), "D").replace(chr(273), "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()

def _cleanup_context_labels(answer: str) -> str:
    """
    Remove internal retrieval labels that can leak into the demo answer.
    Examples: "trong Tài liệu 1", "[Tài liệu 2]", "Ký hiệu 7.14".
    """
    out = answer or ""
    out = re.sub(r"\s*\[\s*Tài\s+liệu\s+\d+\s*\]", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+trong\s+Tài\s+liệu\s+\d+", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*Tài\s+liệu\s+\d+\s*[,;:]?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*Ký\s+hiệu\s+[\w.\-/]+(?:\s+trong\s+Tài\s+liệu\s+\d+)?\s*[,;:]?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\bTheo\s+Điều\s+(\d+[a-zA-ZđĐ]?)\s*,", r"Theo Điều \1,", out, flags=re.IGNORECASE)
    return out.strip()

def _strip_non_yesno_lead(answer: str) -> str:
    """
    Remove accidental LLM yes/no lead from non-yes/no questions.
    Decisions use no-diacritic normalized text to avoid fragile Vietnamese regexes.
    """
    out = answer or ""
    stripped = out.lstrip()
    match = re.match(r"^(\S+)(.*)$", stripped, flags=re.DOTALL)
    if not match:
        return out.strip()

    first_token = match.group(1)
    rest_raw = match.group(2)
    first_norm = _demo_norm(first_token).strip(" ,.:;-")
    rest = rest_raw.lstrip(" ,.:;-")
    rest_norm = _demo_norm(rest).strip()

    if first_norm == "co":
        allowed_prefixes = (
            "cac", "nhung", "hai", "ba", "bon", "nam", "mot",
            "doanh nghiep", "cong ty", "ho kinh doanh", "nguoi su dung", "nguoi lao dong",
        )
        had_separator = first_token.rstrip().endswith((",", ".", ":", ";", "-")) or rest_raw[:1] in {",", ".", ":", ";", "-"}
        if had_separator or rest_norm.startswith(allowed_prefixes):
            return rest.strip()

    if first_norm == "khong":
        citation_prefixes = ("theo", "can cu", "tai", "dieu", "neu", "truong hop")
        had_separator = first_token.rstrip().endswith((",", ".", ":", ";", "-")) or rest_raw[:1] in {",", ".", ":", ";", "-"}
        if had_separator and rest_norm.startswith(citation_prefixes):
            return rest.strip()

    return out.strip()

def normalize_answer_lead(question: str, answer: str) -> str:
    """
    Demo-safe answer cleanup:
    - remove retrieval labels such as "T?i li?u 1" and "K? hi?u 7.14";
    - yes/no question: normalize "Kh?ng Theo..." -> "Kh?ng. Theo...";
    - non-yes/no question: remove accidental "C?," / "C? ..." lead.
    """
    answer = clean_answer(answer)
    answer = _cleanup_context_labels(answer)
    if not answer:
        return answer

    bare_yesno_before_citation = r"^(?P<lead>c?|co|kh?ng|khong)\s*(?:[,.:;-]\s*)?(?=(theo|c?n c?|t?i|?i?u|n?u|tr??ng h?p)\b)"

    if _is_yes_no_question(question):
        def punctuate(m):
            lead = m.group("lead")
            if lead.lower() == "co":
                lead = "C?"
            elif lead.lower() == "khong":
                lead = "Kh?ng"
            return lead[:1].upper() + lead[1:] + ". "
        answer = re.sub(bare_yesno_before_citation, punctuate, answer, count=1, flags=re.IGNORECASE).strip()
        answer = _cleanup_context_labels(answer)
        return answer[:1].upper() + answer[1:] if answer else answer

    answer = _strip_non_yesno_lead(answer)
    answer = _cleanup_context_labels(answer)
    return answer[:1].upper() + answer[1:] if answer else answer

# Extracted from notebook cell 21
import json

import re

import collections

from pathlib import Path

DOC_NAME_MAP_PATH = Path("data/processed/doc_name_map_fixed.json")

_CODE_RE = r"(?:\d+[a-zA-Z]?/\d{4}/[A-ZĐ0-9\-]+|\d+/[A-ZĐ0-9\-]+)"

_TYPES_RE = (
    r"(?:Bộ\s+luật|Luật|Nghị\s+định|Nghị\s+quyết|Pháp\s+lệnh"
    r"|Thông\s+tư\s+liên\s+tịch|Thông\s+tư|Quyết\s+định|Lệnh)"
)

_STOP = (
    rf"(?=\s+(?:ban\s+hành\s+)?ngày\s+\d{{1,2}}/\d{{1,2}}/\d{{4}}"
    rf"|,?\s*có\s+hiệu\s+lực"
    rf"|,?\s*(?:được|bị)\s+(?:sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ)"
    rf"|\s+của\s+(?:Quốc\s+hội|Chính\s+phủ|Bộ|Thủ\s+tướng|Ủy\s+ban)"
    rf"|\s*\)|$)"
)

_PAT_A = re.compile(rf"({_TYPES_RE})\s+số\s+({_CODE_RE})\s+(.+?){_STOP}", flags=re.IGNORECASE)

_PAT_B = re.compile(
    rf"({_TYPES_RE})\s+((?:(?!\bsố\b|\bngày\b|[,)\(]).)+?)\s+số\s+({_CODE_RE})",
    flags=re.IGNORECASE)

_PAT_BARE = re.compile(rf"({_TYPES_RE})\s+(?:số\s+)?({_CODE_RE})", flags=re.IGNORECASE)

_BAD_START = re.compile(
    r"(?i)^(có hiệu|được sửa|được bổ sung|sửa đổi|bổ sung|bị bãi bỏ"
    r"|hết hiệu lực|được thay thế|bị thay thế|và\b|của\b|này\b|số\b)")

def _norm_ws(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _ok_title(t):
    if not t or not (3 < len(t) < 250) or _BAD_START.match(t):
        return False
    # mã văn bản bên trong title chỉ hợp lệ ở dạng "<Loại> số <mã>"
    # (vd: "Hướng dẫn một số điều của Nghị định số 80/2021/NĐ-CP")
    for m in re.finditer(_CODE_RE, t):
        before = t[:m.start()].rstrip()
        if not re.search(rf"(?i){_TYPES_RE}\s+số$", before):
            return False
    return True

def build_doc_name_map(docs):
    """Scan whole corpus once -> {code: full_name}."""
    title_votes = collections.defaultdict(collections.Counter)
    type_votes = collections.defaultdict(collections.Counter)

    for d in docs:
        for t in (_norm_ws(d.get("source_note")), _norm_ws(d.get("related_note"))):
            if not t:
                continue
            for m in _PAT_BARE.finditer(t):
                type_votes[m.group(2)][_norm_ws(m.group(1))] += 1
            for m in _PAT_A.finditer(t):
                ti = m.group(3).strip(" ,;.")
                if _ok_title(ti):
                    title_votes[m.group(2)][ti] += 1
            for m in _PAT_B.finditer(t):
                ti = m.group(2).strip(" ,;.")
                if _ok_title(ti):
                    title_votes[m.group(3)][ti] += 1

    name_map = {}
    for code, tv in type_votes.items():
        loai = tv.most_common(1)[0][0]
        loai = loai[0].upper() + loai[1:].lower()
        if loai.lower() == "bộ luật":
            loai = "Bộ luật"
        if code in title_votes:
            # most votes first; among ties prefer the SHORTER (run-on safe) title
            title = max(title_votes[code].items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
            title = title[0].upper() + title[1:]
            # tránh lặp loại nếu trích yếu đã bắt đầu bằng loại văn bản
            if re.match(rf"^{_TYPES_RE}\b", title, flags=re.IGNORECASE):
                name_map[code] = title
            else:
                name_map[code] = f"{loai} {title}"
        else:
            # fallback: không tìm được trích yếu ở bất kỳ đâu trong corpus
            name_map[code] = f"{loai} số {code}"
    return name_map

def _first_legal_ref_text(doc):
    """Return the best text span for extracting a legal document code/name."""
    for field in (
        "source_note",
        "doc_code",
        "article_title",
        "subject_title",
        "doc_name",
        "source",
        "bm25_search",
        "content",
    ):
        text = _norm_ws(doc.get(field))
        if re.search(_CODE_RE, text or ""):
            return text
    return ""

def _trim_amendment_tail(text):
    # Use unicode escapes so this notebook remains robust across PowerShell encodings.
    phrases = (
        "được sửa đổi",
        "được bổ sung",
        "sửa đổi, bổ sung",
        "được thay thế",
        "bị thay thế",
        "bị bãi bỏ",
        "hết hiệu lực",
    )
    lower = text.lower()
    for phrase in phrases:
        idx = lower.find(phrase)
        if idx != -1:
            return text[:idx]
    return text

DOC_NAME_OVERRIDES = {
    "04/2017/QH14": "Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "50/2005/QH11": "Luật Sở hữu trí tuệ",
    "07/2022/QH15": "Luật sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ",
    "131/2025/QH15": "Luật sửa đổi, bổ sung một số điều của Luật Sở hữu trí tuệ",
    "45/2019/QH14": "Bộ luật Lao động",
    "59/2020/QH14": "Luật Doanh nghiệp",
    "38/2019/QH14": "Luật Quản lý thuế",
    "41/2024/QH15": "Luật Bảo hiểm xã hội",
    "58/2014/QH13": "Luật Bảo hiểm xã hội",
    "36/2005/QH11": "Luật Thương mại",
    "88/2015/QH13": "Luật Kế toán",
    "19/2023/QH15": "Luật Bảo vệ quyền lợi người tiêu dùng",
    "43/2013/QH13": "Luật Đấu thầu",
    "67/2014/QH13": "Luật Đầu tư",
    "80/2021/NĐ-CP": "Nghị định Quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "38/2018/NĐ-CP": "Nghị định Quy định chi tiết về đầu tư cho doanh nghiệp nhỏ và vừa khởi nghiệp sáng tạo",
    "39/2018/NĐ-CP": "Nghị định Quy định chi tiết một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
    "39/2019/NĐ-CP": "Nghị định Về tổ chức và hoạt động của Quỹ Phát triển doanh nghiệp nhỏ và vừa",
    "55/2019/NĐ-CP": "Nghị định Về hỗ trợ pháp lý cho doanh nghiệp nhỏ và vừa",
    "57/2018/NĐ-CP": "Nghị định Về cơ chế, chính sách khuyến khích doanh nghiệp đầu tư vào nông nghiệp, nông thôn",
    "123/2020/NĐ-CP": "Nghị định Quy định về hóa đơn, chứng từ",
    "125/2020/NĐ-CP": "Nghị định Quy định xử phạt vi phạm hành chính về thuế, hóa đơn",
    "102/2021/NĐ-CP": "Nghị định Sửa đổi, bổ sung một số điều của các nghị định về xử phạt vi phạm hành chính trong lĩnh vực thuế, hóa đơn",
    "12/2022/NĐ-CP": "Nghị định Quy định xử phạt vi phạm hành chính trong lĩnh vực lao động, bảo hiểm xã hội",
    "13/2023/NĐ-CP": "Nghị định Về bảo vệ dữ liệu cá nhân",
    "65/2023/NĐ-CP": "Nghị định Quy định chi tiết một số điều và biện pháp thi hành Luật Sở hữu trí tuệ về sở hữu công nghiệp",
    "99/2013/NĐ-CP": "Nghị định Quy định xử phạt vi phạm hành chính trong lĩnh vực sở hữu công nghiệp",
    "126/2021/NĐ-CP": "Nghị định Sửa đổi, bổ sung một số điều của các nghị định xử phạt vi phạm hành chính trong lĩnh vực sở hữu công nghiệp",
    "115/2015/NĐ-CP": "Nghị định Quy định chi tiết một số điều của Luật Bảo hiểm xã hội về bảo hiểm xã hội bắt buộc",
    "143/2018/NĐ-CP": "Nghị định Quy định chi tiết Luật Bảo hiểm xã hội và Luật An toàn, vệ sinh lao động về bảo hiểm xã hội bắt buộc đối với người lao động nước ngoài",
    "28/2015/NĐ-CP": "Nghị định Quy định chi tiết thi hành một số điều của Luật Việc làm về bảo hiểm thất nghiệp",
    "174/2016/NĐ-CP": "Nghị định Quy định chi tiết một số điều của Luật Kế toán",
    "41/2018/NĐ-CP": "Nghị định Quy định xử phạt vi phạm hành chính trong lĩnh vực kế toán, kiểm toán độc lập",
    "78/2021/TT-BTC": "Thông tư Hướng dẫn thực hiện một số điều của Luật Quản lý thuế và Nghị định số 123/2020/NĐ-CP về hóa đơn, chứng từ",
    "80/2021/TT-BTC": "Thông tư Hướng dẫn thi hành một số điều của Luật Quản lý thuế và Nghị định số 126/2020/NĐ-CP",
    "132/2018/TT-BTC": "Thông tư Hướng dẫn chế độ kế toán cho doanh nghiệp siêu nhỏ",
    "133/2016/TT-BTC": "Thông tư Hướng dẫn chế độ kế toán doanh nghiệp nhỏ và vừa",
}

def _canonical_doc_name(code, name=""):
    code = _norm_ws(code)
    if code in DOC_NAME_OVERRIDES:
        return DOC_NAME_OVERRIDES[code]
    name = _norm_ws(name)
    name = re.sub(r"\s+số\s+" + re.escape(code) + r".*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+(?:áp dụng|toàn văn|mới nhất).*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+20\d{2}\s*$", "", name).strip(" -_,.;")
    return name

def extract_doc_code_name(doc):
    """
    Extract stable competition citation keys with canonical names.
    Avoid SEO tails from web titles such as 'áp dụng 2025 mới nhất'.
    """
    if doc.get("doc_code") and doc.get("doc_name"):
        code = _norm_ws(doc.get("doc_code"))
        return code, _canonical_doc_name(code, doc.get("doc_name"))

    text = _trim_amendment_tail(_first_legal_ref_text(doc))
    if not text:
        return "", ""

    m = _PAT_BARE.search(text)
    if m:
        code = m.group(2)
        return code, _canonical_doc_name(code, DOC_NAME_MAP.get(code, f"{m.group(1)} số {code}"))

    m_code = re.search(_CODE_RE, text)
    if not m_code:
        return "", ""

    code = m_code.group(0)
    explicit_name = _norm_ws(doc.get("doc_name") or doc.get("subject_title"))
    if explicit_name and not re.fullmatch(_CODE_RE, explicit_name):
        return code, _canonical_doc_name(code, explicit_name)
    return code, _canonical_doc_name(code, DOC_NAME_MAP.get(code, f"Văn bản số {code}"))

# Extracted from notebook cell 22
import re

from typing import Dict, List

_EV_STOPWORDS = {
    "la", "co", "khong", "duoc", "phai", "cua", "cho", "khi", "thi", "va",
    "ve", "trong", "theo", "neu", "nhu", "nao", "gi", "cac", "nhung",
    "mot", "hai", "ba", "bon", "nam", "do", "nay", "doanh", "nghiep",
    "cong", "ty", "can", "hoi", "quy", "dinh", "phap", "luat", "dieu",
}

_EV_SINGLE_PATTERNS = [
    r"\bco duoc\b", r"\bco phai\b", r"\bco can\b", r"\bduoc khong\b",
    r"\bphai khong\b", r"\bla gi\b", r"\bla ai\b", r"\bbao lau\b",
    r"\bbao nhieu\b", r"\bmuc nao\b", r"\bmuc phat nao\b", r"\bco quan nao\b",
    r"\bai co tham quyen\b", r"\bthoi han\b", r"\bthoi diem nao\b", r"\btu khi nao\b",
]

_EV_MULTI_PATTERNS = [
    r"\b(cac|nhung)\b",
    r"\b(dieu kien|thu tuc|ho so|trach nhiem|nghia vu)\b",
    r"\b(bao gom|gom|dong thoi|ngoai ra|xu ly|khac phuc)\b",
    r"\b(cach nao|nhu the nao|noi dung gi|nhung noi dung gi)\b",
    r"\b(cac noi dung|nhung noi dung|cac cach|nhung cach)\b",
    r"\b(muc ho tro|hinh thuc|truong hop|can cu|quy trinh)\b",
]

_EV_ANSWER_MULTI_PATTERNS = [
    r"\bbao gom\b", r"\bgom\b", r"\bdong thoi\b", r"\bngoai ra\b",
    r"\bmot trong\b", r"\bthoa man\b", r"\bphai\b.*\bva\b",
    r"(?:^|[\s;])(?:1|2|3)[\.\)]\s", r";",
]

_EV_DOMAIN_KEYWORDS = {
    "sme": [
        "doanh nghiep nho va vua", "dnnvv", "sieu nho", "khoi nghiep sang tao",
        "uom tao", "khu lam viec chung", "cum lien ket", "chuoi gia tri",
        "quy phat trien doanh nghiep nho va vua", "mat bang san xuat",
    ],
    "tax": [
        "thue", "hoa don", "chung tu", "bien lai", "ma so thue", "khai thue",
        "nop thua", "hoan thue", "ke toan", "doanh thu tinh thue",
    ],
    "labor": [
        "lao dong", "hop dong lao dong", "nhan vien", "nguoi lao dong", "luong",
        "bao hiem xa hoi", "bhxh", "sa thai", "dinh cong", "nghi viec",
        "bang cap", "lam them", "ca dem", "an toan ve sinh lao dong",
    ],
    "ip": [
        "so huu tri tue", "sang che", "nhan hieu", "kieu dang", "ban quyen",
        "quyen tac gia", "xam pham", "so huu cong nghiep",
    ],
    "procurement": [
        "dau thau", "nha thau", "mua sam", "cptpp", "evfta", "ukvfta",
    ],
    "enterprise": [
        "cong ty", "doanh nghiep", "co dong", "cong ty co phan",
        "dang ky doanh nghiep", "giai the", "pha san", "hop dong",
        "cong ty me", "nguoi dai dien",
    ],
}

_EV_DOMAIN_CODES = {
    "sme": {
        "04/2017/QH14", "80/2021/NĐ-CP", "39/2018/NĐ-CP", "39/2019/NĐ-CP",
        "06/2022/TT-BKHĐT", "05/2019/TT-BKHĐT", "38/2018/NĐ-CP",
        "55/2019/NĐ-CP", "07/2020/TT-BKHCN", "21/2008/QH12",
    },
    "tax": {
        "38/2019/QH14", "126/2020/NĐ-CP", "80/2021/TT-BTC",
        "123/2020/NĐ-CP", "78/2021/TT-BTC", "125/2020/NĐ-CP",
        "88/2015/QH13", "133/2016/TT-BTC", "88/2021/TT-BTC",
    },
    "labor": {
        "45/2019/QH14", "145/2020/NĐ-CP", "12/2022/NĐ-CP",
        "58/2014/QH13", "115/2015/NĐ-CP", "143/2018/NĐ-CP",
        "41/2024/QH15", "84/2015/QH13",
    },
    "ip": {
        "50/2005/QH11", "65/2023/NĐ-CP", "23/2023/TT-BKHCN",
        "01/2007/TT-BKHCN", "99/2013/NĐ-CP", "07/2022/QH15",
    },
    "procurement": {
        "43/2013/QH13", "22/2023/QH15", "95/2020/NĐ-CP", "09/2022/NĐ-CP",
    },
    "enterprise": {
        "59/2020/QH14", "01/2021/NĐ-CP", "122/2021/NĐ-CP",
        "36/2005/QH11", "91/2015/QH13", "19/2023/QH15",
        "54/2010/QH12", "67/2014/QH13",
    },
}

_EV_GENERIC_OK_CODES = {"04/2021/NĐ-CP", "118/2021/NĐ-CP", "15/2022/NĐ-CP"}

_EV_GENERIC_TITLE_PATTERNS = [
    r"\bhieu luc\b", r"\bdieu khoan thi hanh\b", r"\btrach nhiem thi hanh\b",
    r"\bto chuc thuc hien\b", r"\bkhen thuong\b", r"\bxu ly vi pham\b",
    r"\bdieu khoan chuyen tiep\b", r"\bpham vi dieu chinh\b",
]

_EV_WEAK_ACTIONS = {
    "quy dinh", "phap luat", "thuc hien", "co lien quan", "nha nuoc",
}

_EV_ACTION_WORDS = {
    "ho tro", "mien", "giam", "dang ky", "khai", "nop", "hoan", "xu phat",
    "tam dung", "tam dinh chi", "sa thai", "cham dut", "bao cao", "thong bao",
    "cap", "thu hoi", "gia han", "vay", "bao lanh", "lai suat", "huy", "sua doi",
    "bo sung", "khieu nai", "khoi kien", "chuyen doi", "giai the", "tach", "sap nhap",
}

_EV_ENTITY_WORDS = {
    "doanh nghiep nho va vua", "dnnvv", "sieu nho", "cong ty", "nguoi lao dong",
    "nguoi su dung lao dong", "co quan thue", "bo tai chinh", "chu dau tu",
    "nha thau", "co dong", "hoi dong quan tri", "sang che", "nhan hieu",
    "hoa don dien tu", "bien lai dien tu", "bao hiem xa hoi",
}

def _ev_strip_vietnamese(text: str) -> str:
    src = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    dst = "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    return (text or "").translate(str.maketrans(src + src.upper(), dst + dst.upper()))

def _ev_norm(text: str) -> str:
    return _ev_strip_vietnamese(text or "").lower()

def _ev_has_any(text: str, patterns) -> bool:
    t = _ev_norm(text)
    return any(re.search(pattern, t) for pattern in patterns)

def _ev_tokens(text: str) -> set:
    toks = set(re.findall(r"[a-zA-Z0-9đĐ]+", _ev_norm(text)))
    return {t for t in toks if len(t) >= 3 and t not in _EV_STOPWORDS}

def _ev_numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)?", _ev_norm(text)))

def _ev_detect_domains(question: str):
    q = _ev_norm(question)
    domains = []
    for domain, keywords in _EV_DOMAIN_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            domains.append(domain)
    if "sme" in domains and "enterprise" in domains:
        domains.remove("enterprise")
    return domains

def _ev_code_domain_ok(code: str, domains) -> bool:
    if not domains:
        return True
    if code in _EV_GENERIC_OK_CODES:
        return True
    return any(code in _EV_DOMAIN_CODES.get(domain, set()) for domain in domains)

def _ev_article_num(article: str) -> str:
    m = re.search(r"(?:Điều|Dieu)\s+(\d+[a-zA-ZđĐ]?)", article or "", flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""

def _ev_hit_identity(hit):
    doc = hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit
    code, name = extract_doc_code_name(doc)
    article = extract_article_ref(doc)
    article_num = _ev_article_num(article)
    doc_key = f"{code}|{name}" if code and name else ""
    return doc, code or "", name or "", article or "", article_num or "", doc_key

def _ev_mentioned_signals(answer: str):
    answer_norm = _ev_norm(answer)
    codes = set(re.findall(r"\b\d{1,4}/\d{4}/[A-ZĐ0-9-]+\b", answer or "", flags=re.IGNORECASE))
    codes = {c.upper().replace("ND-CP", "NĐ-CP") for c in codes}
    articles = set(re.findall(r"(?:điều|dieu)\s+(\d+[a-zA-ZđĐ]?)", answer_norm, flags=re.IGNORECASE))
    return codes, {a.lower() for a in articles}

def _ev_overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))

def _ev_target_count(question: str, answer: str) -> int:
    q_single = _ev_has_any(question, _EV_SINGLE_PATTERNS)
    q_multi = _ev_has_any(question, _EV_MULTI_PATTERNS)
    a_multi = _ev_has_any(answer, _EV_ANSWER_MULTI_PATTERNS)
    if q_single and not q_multi and not a_multi:
        return 2
    if q_multi and a_multi:
        return 3
    return 2

def _ev_phrase_hits(text: str, phrases) -> set:
    t = _ev_norm(text)
    return {phrase for phrase in phrases if phrase in t}

def _ev_is_generic_title(title: str) -> bool:
    return _ev_has_any(title, _EV_GENERIC_TITLE_PATTERNS)

def _ev_specificity_score(question: str, answer: str, text: str) -> float:
    """Generic direct-evidence signal: entities, actions, and numbers must line up."""
    qa = f"{question} {answer}"
    qa_actions = _ev_phrase_hits(qa, _EV_ACTION_WORDS)
    tx_actions = _ev_phrase_hits(text, _EV_ACTION_WORDS)
    qa_entities = _ev_phrase_hits(qa, _EV_ENTITY_WORDS)
    tx_entities = _ev_phrase_hits(text, _EV_ENTITY_WORDS)
    qa_weak = _ev_phrase_hits(qa, _EV_WEAK_ACTIONS)
    tx_weak = _ev_phrase_hits(text, _EV_WEAK_ACTIONS)

    action_score = len(qa_actions & tx_actions) / max(1, min(3, len(qa_actions)))
    entity_score = len(qa_entities & tx_entities) / max(1, min(3, len(qa_entities)))
    weak_score = 0.05 * len(qa_weak & tx_weak)
    return min(1.0, 0.48 * action_score + 0.42 * entity_score + weak_score)

def _ev_candidate_score(hit, rank: int, question: str, answer: str, domains, answer_codes, answer_articles):
    doc, code, name, article, article_num, doc_key = _ev_hit_identity(hit)
    title = doc.get("article_title", "")
    text = " ".join([
        title,
        doc.get("doc_name", ""),
        doc.get("subject_title", ""),
        doc.get("content", ""),
    ])
    q_tokens = _ev_tokens(question)
    a_tokens = _ev_tokens(answer)
    title_tokens = _ev_tokens(title)
    content_tokens = _ev_tokens(text)
    q_nums = _ev_numbers(question)
    a_nums = _ev_numbers(answer)
    c_nums = _ev_numbers(text)

    rerank_score = float(hit.get("rerank_score", 0.0))
    domain_ok = _ev_code_domain_ok(code, domains)
    mentioned = (code.upper() in answer_codes) or (article_num and article_num in answer_articles)
    generic_title = _ev_is_generic_title(title)

    q_content = _ev_overlap(q_tokens, content_tokens)
    q_title = _ev_overlap(q_tokens, title_tokens)
    a_content = _ev_overlap(a_tokens, content_tokens)
    num_overlap = _ev_overlap(q_nums | a_nums, c_nums)
    specificity = _ev_specificity_score(question, answer, text)

    score = 0.0
    score += 1.00 * rerank_score
    score += 0.26 * q_content
    score += 0.18 * q_title
    score += 0.12 * a_content
    score += 0.14 * num_overlap
    score += 0.20 * specificity
    score += 0.18 if domain_ok else -0.14
    score += 0.22 if mentioned else 0.0
    score -= 0.05 if generic_title and not mentioned else 0.0
    score -= 0.030 * max(0, rank - 1)

    evidence = (
        0.32 * q_content
        + 0.22 * q_title
        + 0.18 * a_content
        + 0.16 * num_overlap
        + 0.22 * specificity
        + (0.16 if mentioned else 0.0)
        + (0.08 if domain_ok else -0.12)
        - (0.08 if generic_title and not mentioned else 0.0)
    )

    return {
        "hit": hit,
        "rank": rank,
        "score": score,
        "evidence": evidence,
        "rerank_score": rerank_score,
        "code": code,
        "article": article,
        "article_num": article_num,
        "doc_key": doc_key,
        "domain_ok": domain_ok,
        "mentioned": mentioned,
        "generic_title": generic_title,
    }

def _ev_add_threshold(position: int, target: int, multi: bool) -> float:
    """Minimum direct-evidence needed for the next output slot."""
    if position <= 2:
        return 0.34 if multi else 0.40
    if position == 3:
        return 0.54 if multi else 0.80
    return 1.00

def _ev_accept_candidate(cand, selected, top_score: float, target: int, multi: bool) -> bool:
    position = len(selected) + 1
    if cand["mentioned"]:
        if position >= 3:
            return multi and cand["rank"] <= 3 and cand["evidence"] >= 0.48
        return cand["rank"] <= 4 and cand["evidence"] >= 0.30
    if cand["generic_title"] and cand["evidence"] < 0.56:
        return False
    if not cand["domain_ok"] and cand["evidence"] < 0.52:
        return False
    if cand["rank"] >= 5:
        return False
    if cand["rank"] >= 4 and cand["evidence"] < 0.56:
        return False

    min_evidence = _ev_add_threshold(position, target, multi)
    if cand["evidence"] < min_evidence:
        return False

    # Article 2 must be close; article 3 must be extremely direct.
    allowed_gap = 0.11 if position <= 2 else 0.06
    if top_score - cand["score"] > allowed_gap and cand["evidence"] < 0.62:
        return False
    if position >= 3 and not (multi and cand["rank"] <= 3 and cand["evidence"] >= 0.54 and (cand["domain_ok"] or cand["mentioned"])):
        return False
    return True

def select_output_hits_v4(answer: str, kept: List[Dict], max_articles: int = 4, fallback_articles: int = 1, question: str = "") -> List[Dict]:
    """
    Evidence verifier over top8 reranked candidates.

    It keeps top8 recall as the candidate bank, then only outputs articles with
    enough direct evidence. No labels, no id rules, no external/closed models.
    """
    unique = []
    seen_article = set()
    for hit in kept:
        doc, code, name, article, article_num, doc_key = _ev_hit_identity(hit)
        if not code or not name or not article:
            continue
        key = (code, article_num or article.lower())
        if key in seen_article:
            continue
        seen_article.add(key)
        unique.append(hit)

    if not unique:
        return kept[:fallback_articles]

    domains = _ev_detect_domains(question)
    answer_codes, answer_articles = _ev_mentioned_signals(answer)
    target = min(max_articles, _ev_target_count(question, answer), len(unique))
    q_multi = _ev_has_any(question, _EV_MULTI_PATTERNS)
    a_multi = _ev_has_any(answer, _EV_ANSWER_MULTI_PATTERNS)
    multi = q_multi or a_multi
    if not multi:
        target = min(target, 2)
    max_docs = 2 if multi else 1
    max_articles_per_doc = 2 if multi else 1

    candidates = [
        _ev_candidate_score(hit, i + 1, question, answer, domains, answer_codes, answer_articles)
        for i, hit in enumerate(unique[:8])
    ]

    # Always seed with rank 1: it is usually the highest-recall anchor.
    selected = [candidates[0]]
    selected_keys = {(candidates[0]["code"], candidates[0]["article_num"] or candidates[0]["article"])}
    selected_docs = {candidates[0]["doc_key"]}
    top_score = candidates[0]["score"]

    # Prefer high evidence. Keep rank as a tie-breaker to avoid overfitting to noisy lexical overlap.
    ordered = sorted(candidates[1:], key=lambda c: (c["score"], -c["rank"]), reverse=True)

    for cand in ordered:
        if len(selected) >= target:
            break
        key = (cand["code"], cand["article_num"] or cand["article"])
        if key in selected_keys:
            continue

        doc_count = sum(1 for x in selected if x["doc_key"] == cand["doc_key"])
        if doc_count >= max_articles_per_doc:
            continue
        if cand["doc_key"] not in selected_docs and len(selected_docs) >= max_docs:
            continue

        if not _ev_accept_candidate(cand, selected, top_score, target, multi):
            continue

        selected.append(cand)
        selected_keys.add(key)
        selected_docs.add(cand["doc_key"])

    # Very limited backfill: only if top2/top3 has direct evidence. This keeps
    # top1 guaranteed while making article 2 intentional and article 3 rare.
    if multi and len(selected) < min(2, target):
        for cand in candidates[1:3]:
            key = (cand["code"], cand["article_num"] or cand["article"])
            if key in selected_keys:
                continue
            if cand["evidence"] < 0.34 and not cand["mentioned"]:
                continue
            selected.append(cand)
            selected_keys.add(key)
            if len(selected) >= min(2, target):
                break

    selected.sort(key=lambda c: c["rank"])
    return [c["hit"] for c in selected[:max(1, min(target, len(selected)))]]

# Extracted from notebook cell 23
import re

from functools import lru_cache

from typing import Dict, List, Tuple

_V4_SELECT_OUTPUT_HITS = select_output_hits_v4

_V5_HARD_SINGLE_PATTERNS = [
    r"\bbao lau\b", r"\bbao nhieu\b", r"\bthoi han\b", r"\bthoi diem\b",
    r"\btu khi nao\b", r"\bla gi\b", r"\bdinh nghia\b", r"\bhieu luc\b",
    r"\bco quan nao\b", r"\bai co tham quyen\b", r"\bmuc nao\b", r"\bmuc phat nao\b",
]

_V5_STRONG_MULTI_PATTERNS = [
    r"\b(cac|nhung)\b",
    r"\bbao gom\b", r"\bgom\b", r"\bdong thoi\b", r"\bngoai ra\b",
    r"\b(nhung noi dung|cac noi dung|nhung cach|cac cach)\b",
    r"\b(noi dung va muc|muc ho tro|hinh thuc ho tro)\b",
    r"\b(cac truong hop|nhung truong hop|truong hop nao)\b",
    r"\b(xu ly).*\b(khac phuc)\b",
]

_V5_BAD_TITLE_PATTERNS = [
    r"\bhieu luc\b", r"\bdieu khoan thi hanh\b", r"\btrach nhiem thi hanh\b",
    r"\bto chuc thuc hien\b", r"\bkhen thuong\b", r"\bdieu khoan chuyen tiep\b",
]

def _v5_valid_doc_code(code: str) -> bool:
    # Typical competition document code contains a 4-digit year and a legal-doc suffix.
    return bool(re.match(r"^\d{2,4}/\d{4}/[A-ZĐ0-9-]+$", (code or "").upper()))

def _v5_has_any(text: str, patterns) -> bool:
    t = _ev_norm(text)
    return any(re.search(pattern, t) for pattern in patterns)

def _v5_answer_mentions(answer: str) -> Tuple[set, set]:
    return _ev_mentioned_signals(answer)

def _v5_citation_key(hit) -> str:
    doc, code, name, article, article_num, doc_key = _ev_hit_identity(hit)
    if not code or not name or not article:
        return ""
    return f"{code}|{name}|{article}"

def _v5_hit_rank(hit, kept: List[Dict]) -> int:
    for idx, item in enumerate(kept, start=1):
        if item is hit:
            return idx
    key = _v5_citation_key(hit)
    for idx, item in enumerate(kept, start=1):
        if _v5_citation_key(item) == key:
            return idx
    return 99

def _v5_doc_text(hit, max_chars: int) -> str:
    doc = hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit
    parts = [
        doc.get("article_title", ""),
        doc.get("doc_name", ""),
        doc.get("subject_title", ""),
        doc.get("content", "") or doc.get("bm25_search", ""),
    ]
    text = "\n".join(p for p in parts if p)
    return text[:max_chars]

def _v5_suspicion_reason(hit, kept: List[Dict], answer: str, question: str, domains) -> str:
    doc, code, name, article, article_num, doc_key = _ev_hit_identity(hit)
    rank = _v5_hit_rank(hit, kept)
    answer_codes, answer_articles = _v5_answer_mentions(answer)
    mentioned = (code.upper() in answer_codes) or (article_num and article_num in answer_articles)
    score = float(hit.get("rerank_score", 0.0))
    title = doc.get("article_title", "")
    hard_single = _v5_has_any(question, _V5_HARD_SINGLE_PATTERNS)
    strong_multi = _v5_has_any(question, _V5_STRONG_MULTI_PATTERNS)

    if not _v5_valid_doc_code(code):
        return "invalid_doc_code"
    if rank == 1:
        return ""
    if mentioned:
        return ""
    if hard_single and not strong_multi:
        return "hard_single_extra"
    if _v5_has_any(title, _V5_BAD_TITLE_PATTERNS):
        return "generic_title"
    if domains and not _ev_code_domain_ok(code, domains) and score < 0.88:
        return "off_domain"
    if rank >= 4 and score < 0.75:
        return "late_low_score"
    return ""

def _v5_llm_verdict_cached(question: str, answer: str, citation: str, text: str) -> str:
    if not CONFIG.get("citation_verifier_enabled", True):
        return "UNKNOWN"
    try:
        prompt = (
            "Bạn là bộ lọc trích dẫn pháp luật. Nhiệm vụ: xác định điều luật dưới đây "
            "có nên xuất hiện trong relevant_articles cho câu hỏi không.\n"
            "Chỉ trả lời đúng một từ: DIRECT, SUPPORT, hoặc IRRELEVANT.\n\n"
            "DIRECT: điều luật trực tiếp chứa căn cứ trả lời câu hỏi.\n"
            "SUPPORT: điều luật liên quan/hỗ trợ, chỉ cần giữ nếu câu hỏi cần nhiều căn cứ.\n"
            "IRRELEVANT: điều luật chỉ cùng chủ đề hoặc không cần để trả lời.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời hệ thống: {answer[:900]}\n\n"
            f"Trích dẫn ứng viên: {citation}\n"
            f"Nội dung điều luật:\n{text}\n\n"
            "Kết luận:"
        )
        resp = ollama.chat(
            model=CONFIG.get("citation_verifier_model", CONFIG.get("llm_model", "qwen2.5:7b")),
            messages=[
                {"role": "system", "content": "Return only DIRECT, SUPPORT, or IRRELEVANT."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": int(CONFIG.get("citation_verifier_num_predict", 6)),
            },
        )
        text_out = (resp.get("message", {}) or {}).get("content", "").strip().upper()
        if "IRRELEVANT" in text_out:
            return "IRRELEVANT"
        if "DIRECT" in text_out:
            return "DIRECT"
        if "SUPPORT" in text_out:
            return "SUPPORT"
    except Exception:
        return "UNKNOWN"
    return "UNKNOWN"

def _v5_llm_verdict(question: str, answer: str, hit) -> str:
    citation = _v5_citation_key(hit)
    text = _v5_doc_text(hit, int(CONFIG.get("citation_verifier_max_content_chars", 1200)))
    return _v5_llm_verdict_cached(question, answer, citation, text)

def _v5_refine_v4_hits(answer: str, kept: List[Dict], v4_hits: List[Dict], question: str) -> List[Dict]:
    if not v4_hits:
        return v4_hits

    domains = _ev_detect_domains(question)
    strong_multi = _v5_has_any(question, _V5_STRONG_MULTI_PATTERNS)
    refined = []

    for idx, hit in enumerate(v4_hits):
        doc, code, name, article, article_num, doc_key = _ev_hit_identity(hit)

        # Top-1 anchor is the recall anchor. Keep it unless the code is structurally invalid
        # and there is another v4 citation available.
        reason = _v5_suspicion_reason(hit, kept, answer, question, domains)
        if idx == 0:
            if reason == "invalid_doc_code" and len(v4_hits) > 1:
                continue
            refined.append(hit)
            continue

        # Cheap deterministic veto for structurally invalid citations.
        if reason == "invalid_doc_code":
            continue

        # Not suspicious: keep v4's choice.
        if not reason:
            refined.append(hit)
            continue

        # Suspicious extras are kept unless the verifier confidently says IRRELEVANT.
        verdict = _v5_llm_verdict(question, answer, hit)
        if verdict == "IRRELEVANT":
            continue
        if (
            verdict == "SUPPORT"
            and CONFIG.get("citation_verifier_drop_support_on_single", False)
            and not strong_multi
            and len(refined) >= 1
        ):
            continue
        refined.append(hit)

    if not refined:
        # Absolute fallback: return the original top v4 citation.
        return v4_hits[:1]

    # Keep v4's article-count discipline: at most 3.
    return refined[: min(3, len(refined))]

def select_output_hits_v5(answer: str, kept: List[Dict], max_articles: int = 3, fallback_articles: int = 1, question: str = "") -> List[Dict]:
    v4_hits = _V4_SELECT_OUTPUT_HITS(
        answer=answer,
        kept=kept,
        max_articles=max_articles,
        fallback_articles=fallback_articles,
        question=question,
    )
    return _v5_refine_v4_hits(answer=answer, kept=kept, v4_hits=v4_hits, question=question)

# Extracted from notebook cell 24
import re

from typing import Dict, List

_V63_BASE_SELECT_OUTPUT_HITS = select_output_hits_v5

_V63_GENERIC_TITLE_PATTERNS = [
    r"\bhieu luc\b",
    r"\bdieu khoan thi hanh\b",
    r"\bto chuc thuc hien\b",
    r"\btrach nhiem thi hanh\b",
    r"\bquy dinh chuyen tiep\b",
    r"\bsua doi\b.*\bbo sung\b",
    r"\bbai bo\b",
    r"\bkhen thuong\b",
]

def _v63_norm(text: str) -> str:
    text = strip_diacritics(str(text or ""))
    text = text.replace("?", "d").replace("?", "D")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def _v63_doc(hit):
    return hit["doc"] if isinstance(hit, dict) and "doc" in hit else hit

def _v63_doc_key(hit) -> str:
    doc = _v63_doc(hit)
    code, name = extract_doc_code_name(doc)
    return f"{code}|{name}" if code and name else ""

def _v63_article_key(hit) -> str:
    doc = _v63_doc(hit)
    code, name = extract_doc_code_name(doc)
    article = extract_article_ref(doc)
    return f"{code}|{name}|{article}" if code and name and article else ""

def _v63_text(hit, max_chars=900) -> str:
    doc = _v63_doc(hit)
    return " ".join([
        str(doc.get("doc_title", "")),
        str(doc.get("doc_name", "")),
        str(doc.get("article_title", "")),
        str(doc.get("chapter_title", "")),
        str(doc.get("source_note", "")),
        str(doc.get("content", ""))[:max_chars],
    ])

def _v63_is_generic_noise(question: str, hit) -> bool:
    q = _v63_norm(question)
    text = _v63_norm(_v63_text(hit))
    if not any(re.search(pattern, text) for pattern in _V63_GENERIC_TITLE_PATTERNS):
        return False
    allow_terms = [
        "hieu luc", "thi hanh", "chuyen tiep", "sua doi", "bai bo",
        "trach nhiem thi hanh", "to chuc thuc hien",
    ]
    return not any(term in q for term in allow_terms)

def _v63_add_same_doc_band(question: str, selected: List[Dict], kept: List[Dict], max_articles: int) -> List[Dict]:
    if len(selected) >= max_articles:
        return selected

    selected_docs = {_v63_doc_key(hit) for hit in selected}
    selected_keys = {_v63_article_key(hit) for hit in selected}
    low = float(CONFIG.get("v63_band_low", 0.90))
    high = float(CONFIG.get("v63_band_high", 0.92))
    max_rank = int(CONFIG.get("v63_band_max_rank", 4))

    for rank, hit in enumerate(kept, start=1):
        if rank > max_rank:
            break
        key = _v63_article_key(hit)
        if not key or key in selected_keys:
            continue
        if _v63_doc_key(hit) not in selected_docs:
            continue
        score = float(hit.get("rerank_score", 0.0))
        if not (low <= score < high):
            continue
        if _v63_is_generic_noise(question, hit):
            continue
        selected.append(hit)
        break
    return selected

def select_output_hits(answer: str, kept: List[Dict], max_articles: int = 3, fallback_articles: int = 1, question: str = "") -> List[Dict]:
    selected = list(_V63_BASE_SELECT_OUTPUT_HITS(
        answer=answer,
        kept=kept,
        max_articles=max_articles,
        fallback_articles=fallback_articles,
        question=question,
    ))
    selected = _v63_add_same_doc_band(question, selected, kept, max_articles)
    return selected[:max_articles]

# Extracted from notebook cell 25
from tqdm.auto import tqdm

from datetime import datetime

from pathlib import Path

import json

import zipfile

def _save_json_atomic(path, data):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)

def _zip_results(exp_dir, results_path):
    with zipfile.ZipFile(exp_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(results_path, arcname="results.json")

def run_experiment(test_data, cfg, limit=None, resume_dir=None, autosave_every=1):
    """
    Run generation with per-record checkpointing.

    If interrupted, rerun with:
        run_experiment(test_data, CONFIG, resume_dir="submission/<folder>")
    It will load results_partial.json/results.json and skip completed ids.
    """
    if resume_dir:
        exp_dir = Path(resume_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = Path("submission") / f"{datetime.now():%Y%m%d_%H%M}_{cfg['exp_name']}"
        exp_dir.mkdir(parents=True, exist_ok=True)

    partial_path = exp_dir / "results_partial.json"
    results_path = exp_dir / "results.json"
    config_path = exp_dir / "config.json"

    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as f:
            results = json.load(f)
        print("Resuming from partial:", partial_path, "records:", len(results))
    elif results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        print("Resuming from final results:", results_path, "records:", len(results))
    else:
        results = []

    _save_json_atomic(config_path, cfg)

    completed_ids = {r.get("id") for r in results}
    data = test_data[:limit] if limit else test_data
    pending = [item for item in data if item.get("id") not in completed_ids]

    print("Output dir:", exp_dir)
    print("Completed:", len(completed_ids), "Pending:", len(pending), "Total target:", len(data))

    try:
        for item in tqdm(pending, desc=cfg["exp_name"]):
            question = item.get("question")

            retrieval = hybrid_dense_bm25_search(
                query=question,
                bm25_top_k=cfg["bm25_top_k"],
                dense_top_k=cfg["dense_top_k"],
                bm25_family_top_k=cfg["bm25_family_top_k"],
                final_fused_top_k=cfg["final_fused_top_k"],
                rrf_k=cfg["rrf_k"],
            )

            if cfg["use_reranker"]:
                kept = rerank_hits(
                    query=question,
                    hits=retrieval["final_fused_hits"],
                    top_k=cfg["final_top_k"],
                    rerank_input_k=cfg["rerank_input_k"],
                    batch_size=cfg["rerank_batch_size"],
                )
            else:
                kept = retrieval["final_fused_hits"][:cfg["final_top_k"]]

            answer = llm.generate(question, kept)
            if cfg.get("strip_extra_dieu"):
                answer = strip_uncited_dieu(answer, kept)
            answer = normalize_answer_lead(question, answer)
            if cfg.get("inject_missing_citations"):
                answer = inject_citations(answer, kept)
            output_kept = select_output_hits(
                answer=answer,
                kept=kept,
                max_articles=cfg.get("max_output_articles", 3),
                fallback_articles=cfg.get("fallback_output_articles", 1),
                question=question,
            )

            results.append(format_competition_output(
                qid=item.get("id"),
                question=question,
                answer=answer,
                kept=output_kept,
            ))
            completed_ids.add(item.get("id"))

            if autosave_every and len(results) % autosave_every == 0:
                _save_json_atomic(partial_path, results)

    except Exception:
        _save_json_atomic(partial_path, results)
        print("Interrupted/error. Saved partial:", partial_path, "records:", len(results))
        raise

    # Preserve input order, useful after resume.
    order = {item.get("id"): idx for idx, item in enumerate(data)}
    results.sort(key=lambda r: order.get(r.get("id"), 10**9))

    _save_json_atomic(partial_path, results)
    _save_json_atomic(results_path, results)
    _zip_results(exp_dir, results_path)

    print("Saved:", exp_dir)
    print("Records:", len(results))
    return results
