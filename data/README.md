# Local artifacts

The full corpus, BM25 pickle and dense matrix are intentionally excluded from Git.
Set LEGAL_DATA_DIR to the existing submission artifact directory. See docs/DATA.md.
Do not load pickle files from arbitrary third parties; the runtime checks the supplied
bundle against configs/artifact_manifest.json before deserializing it.
