# Source provenance and refactor boundaries

Input: `AgenticBuilders_R2AI_Submission_Package (1).rar` supplied by the user.
Original notebook SHA-256:
`e848db9a9765edb90f0d5ce596db75b19d893d7123d57860092f3e06637fe67f`.

The original notebook is retained under `notebooks/submission_pipeline.ipynb`.
`pipeline_core.py` extracts code definitions/constants from cells 3, 5, 12, 13, 14,
17, 19–25. It excludes notebook model/data initialization, prints, intermediate
CONFIG mutations, disk-based document-map initialization and automatic batch launch.
Final settings come from the supplied `v6_3_config.json`. Runtime initialization
supplies document maps, models and the Ollama client explicitly. Notebook global
state is retained in the extracted core, with serialized app access, rather than
claimed to be fully redesigned or thread-safe.

Intentional algorithm-edge fixes:

1. Clamp BM25 top-k to corpus length (and handle zero), preventing small fixtures
   from failing NumPy argpartition when k exceeds document count.
2. When the first article alone exceeds the context budget, retain its truncated
   text rather than silently sending no context. This changes long-context input;
   original submission scores cannot automatically be transferred to this refactor.

New surrounding code: package/config layout, ignored environment configuration,
lazy model initialization, checksummed artifact loading, input validation, explicit
saved-output preview, article display, sanitized errors, single-request guard,
offline tests, Docker preview and a prepared CI workflow.

Preserved limitations include heuristic citation/domain rules, model-generated
answers without formal entailment checking, citation-verifier fail-open behaviour
(the original verifier returns a keep verdict after some exceptions), and no complete
legal-effective-date resolver. The postprocessor may also fall back to a retrieved
article; a selected reference is not a correctness certificate.

No model training/fine-tuning, FAISS, MCP, Copilot Studio, public deployment or full
live-model validation was added. The local sample preview serves six saved outputs;
it does not run retrieval/generation and does not answer arbitrary new questions.

The historical 2,000-question output and reported score were not rescored. Competition
metric names/definitions and public leaderboard evidence were not included. Do not
label the supplied 0.6543 value as accuracy. This code release omits full data,
indexes, model weights and full question/output sets. The six preview examples
are explicitly labelled as archived outputs. Full-data redistribution rights and
end-to-end live-model validation remain outside the scope of this code publication.
