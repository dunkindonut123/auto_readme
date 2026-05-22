"""
readme_engine.py
================
An AST-driven text-extraction and multi-model summarization engine for Python source code.
Supported Algorithms: TF-IDF, BM25, and LSA (comparison models) evaluated
against a Lead Sentence baseline.

All models are trained locally — no pre-trained weights or external downloads required.
Three complementary evaluation metrics are used:
  - ROUGE   (rouge1, rouge2, rougeL)   — n-gram overlap against a reference
  - BERTScore (precision, recall, F1)  — semantic similarity via BERT embeddings
  - Coverage  (reference-free)         — fraction of key file concepts mentioned

The winning model is elected by the metric and scoring function specified at runtime
(default: rougeL). BERTScore requires: pip install bert-score
"""

import ast
import re
import math
import numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# Text Preprocessing & Tokenization
# ══════════════════════════════════════════════════════════════

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "for",
    "with", "that", "this", "as", "be", "are", "was", "were", "by",
    "from", "on", "at", "we", "our", "you", "they", "their", "its",
    "have", "has", "had", "not", "but", "so", "if", "do", "can", "will",
    "return", "returns", "none", "true", "false", "self", "cls",
}


def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip non-alphanumeric characters, and remove stopwords.
    Tokens shorter than 3 characters are discarded to reduce noise.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """
    Split a block of text into individual sentences using punctuation boundaries.
    Sentences shorter than 10 characters are discarded as non-informative fragments.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


# ══════════════════════════════════════════════════════════════
# AST Code Parsing Utility
# ══════════════════════════════════════════════════════════════

def parse_python_file(source: str) -> dict:
    """
    Use Python's native AST to extract accurately scoped module metadata.
    Collects module description, global functions, classes with their methods,
    top-level imports, and a flat docstring candidate pool for summarization.
    """
    metadata = {
        "description": "",
        "functions":   [],   # Global (module-level) functions only
        "classes":     [],   # Classes, each containing their own methods
        "imports":     [],   # Top-level import statements as reconstructed strings
        "docstrings":  [],   # Flat candidate pool fed into the ranking models
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"[Parser Error] Failed parsing syntax: {e}")
        return metadata

    # Module-level docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        metadata["description"] = module_doc.strip()
        metadata["docstrings"].extend(split_sentences(module_doc))

    for node in tree.body:

        # ── Imports ──────────────────────────────────────────
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    metadata["imports"].append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                names = ", ".join(a.name for a in node.names)
                metadata["imports"].append(f"from {node.module} import {names}")

        # ── Global functions ──────────────────────────────────
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            args = [arg.arg for arg in node.args.args if arg.arg != "self"]
            doc  = ast.get_docstring(node) or ""
            metadata["functions"].append({
                "name":      node.name,
                "args":      args,
                "docstring": doc.strip(),
            })
            if doc:
                metadata["docstrings"].extend(split_sentences(doc))

        # ── Classes ───────────────────────────────────────────
        elif isinstance(node, ast.ClassDef):
            class_doc  = ast.get_docstring(node) or ""
            class_item = {
                "name":      node.name,
                "docstring": class_doc.strip(),
                "methods":   [],
            }
            if class_doc:
                metadata["docstrings"].extend(split_sentences(class_doc))

            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                # Keep __init__ for usage-example generation; skip other private methods
                if child.name.startswith("_") and child.name != "__init__":
                    continue
                m_args = [arg.arg for arg in child.args.args if arg.arg != "self"]
                m_doc  = ast.get_docstring(child) or ""
                class_item["methods"].append({
                    "name":      child.name,
                    "args":      m_args,
                    "docstring": m_doc.strip(),
                })
                if m_doc:
                    metadata["docstrings"].extend(split_sentences(m_doc))

            metadata["classes"].append(class_item)

    return metadata


# ══════════════════════════════════════════════════════════════
# Ranking Model 1 — TF-IDF
# ══════════════════════════════════════════════════════════════

class TFIDFModel:
    """
    Classic TF-IDF ranking model trained on a local corpus of docstrings.
    IDF weights are computed from the training corpus; TF is computed per sentence.
    Sentences are scored by the mean of their TF-IDF term weights.
    """

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.corpus_size: int = 0

    def fit(self, corpus: list[str]) -> "TFIDFModel":
        """
        Compute inverse-document-frequency weights from a list of training documents.
        Uses Laplace smoothing to avoid zero-division on unseen terms.
        """
        self.corpus_size = len(corpus)
        df: dict[str, int] = defaultdict(int)
        for doc in corpus:
            for term in set(tokenize(doc)):
                df[term] += 1
        self.idf = {
            t: math.log((self.corpus_size + 1) / (f + 1)) + 1
            for t, f in df.items()
        }
        return self

    def vectorize(self, text: str) -> dict[str, float]:
        """
        Produce a TF-IDF vector (term → weight) for an arbitrary piece of text.
        Unknown terms fall back to an IDF weight of 1.0 so they are not penalised
        for being absent from the training corpus.
        """
        tokens = tokenize(text)
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        n  = len(tokens) or 1
        tf = {t: c / n for t, c in freq.items()}
        return {t: tf[t] * self.idf.get(t, 1.0) for t in tf}

    def score_sentence(self, sentence: str) -> float:
        """
        Return the mean TF-IDF weight across all terms in the sentence.
        Higher scores indicate sentences whose vocabulary is both frequent in the
        sentence and rare in the broader corpus — i.e. informative sentences.
        """
        vec = self.vectorize(sentence)
        return sum(vec.values()) / (len(vec) + 1e-9)

    def summarize(self, sentences: list[str], top_n: int = 2) -> list[str]:
        """
        Return the top_n highest-scoring sentences, preserving their original order.
        Original order is preserved so the summary reads naturally.
        """
        if not sentences:
            return []
        scored   = [(s, self.score_sentence(s)) for s in sentences]
        selected = {s for s, _ in sorted(scored, key=lambda x: -x[1])[:top_n]}
        return [s for s in sentences if s in selected]


# ══════════════════════════════════════════════════════════════
# Ranking Model 2 — BM25
# ══════════════════════════════════════════════════════════════

class BM25Model:
    """
    BM25 Okapi ranking model backed by the rank_bm25 library.
    Uses the library's official get_scores() API instead of manually
    accessing internal attributes, making it robust across library versions.
    Requires: pip install rank-bm25
    """

    def __init__(self):
        self._model = None   # BM25Okapi instance set after fit()

    def fit(self, corpus: list[str]) -> "BM25Model":
        """
        Tokenize the corpus and initialise the BM25Okapi index.
        The index encodes document frequencies and average document length
        needed for BM25's length-normalisation term.
        """
        from rank_bm25 import BM25Okapi
        tokenized = [tokenize(doc) for doc in corpus]
        # BM25Okapi raises if the corpus is completely empty
        if not any(tokenized):
            return self
        self._model = BM25Okapi(tokenized)
        return self

    def summarize(self, sentences: list[str], top_n: int = 2) -> list[str]:
        """
        Score each candidate sentence against the full sentence set as a query,
        then return the top_n highest-scoring sentences in original order.
        Using get_scores() is the stable public API — avoids fragile attribute access.
        """
        if not sentences or self._model is None:
            return []

        # Aggregate all candidate tokens as a single pseudo-query so BM25 can
        # rank each sentence by its relevance to the collective vocabulary.
        all_tokens = tokenize(" ".join(sentences))
        if not all_tokens:
            return sentences[:top_n]

        # Re-create a temporary index over just the candidates so the corpus
        # size matches; then score each sentence against the aggregate query.
        from rank_bm25 import BM25Okapi
        candidate_tokens = [tokenize(s) for s in sentences]
        local_index      = BM25Okapi([t if t else ["<empty>"] for t in candidate_tokens])
        scores           = local_index.get_scores(all_tokens)

        ranked   = sorted(zip(sentences, scores), key=lambda x: -x[1])
        selected = {s for s, _ in ranked[:top_n]}
        return [s for s in sentences if s in selected]


# ══════════════════════════════════════════════════════════════
# Ranking Model 3 — LSA (Latent Semantic Analysis)
# ══════════════════════════════════════════════════════════════

class LSAModel:
    """
    Latent Semantic Analysis summarizer using truncated SVD.
    Builds a term-document matrix from the corpus, decomposes it with SVD to
    obtain a k-dimensional concept space, then scores each candidate sentence
    by the L2 norm of its projection onto that concept space.
    k is capped at the actual rank of the matrix to avoid index errors on
    small corpora.
    """

    def __init__(self, k: int = 25):
        self.k:          int              = k
        self.vocabulary: dict[str, int]  = {}
        self.U_k:        np.ndarray|None = None

    def fit(self, corpus: list[str]) -> "LSAModel":
        """
        Build the term-document matrix and compute the truncated SVD.
        The left singular vectors (U_k) encode the latent concept space.
        k is automatically clamped to min(requested_k, actual_rank) so the
        model degrades gracefully on tiny corpora.
        """
        unique_terms:     set[str]        = set()
        tokenized_corpus: list[list[str]] = []

        for doc in corpus:
            tokens = tokenize(doc)
            tokenized_corpus.append(tokens)
            unique_terms.update(tokens)

        if not unique_terms:
            return self

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(unique_terms))}

        A = np.zeros((len(self.vocabulary), len(corpus)))
        for doc_idx, tokens in enumerate(tokenized_corpus):
            for token in tokens:
                if token in self.vocabulary:
                    A[self.vocabulary[token], doc_idx] += 1

        U, _, _ = np.linalg.svd(A, full_matrices=False)
        effective_k  = min(self.k, U.shape[1])
        self.U_k     = U[:, :effective_k]
        return self

    def summarize(self, sentences: list[str], top_n: int = 2) -> list[str]:
        """
        Project each candidate sentence into the latent concept space and score
        it by the L2 norm of the resulting concept vector.
        Sentences that activate more latent concepts more strongly rank higher.
        """
        if not sentences or self.U_k is None:
            return []

        scores: list[float] = []
        for tokens in (tokenize(s) for s in sentences):
            vec = np.zeros(len(self.vocabulary))
            for t in tokens:
                if t in self.vocabulary:
                    vec[self.vocabulary[t]] += 1
            concept_vec = vec @ self.U_k if np.any(vec) else np.zeros(self.U_k.shape[1])
            scores.append(float(np.linalg.norm(concept_vec)))

        ranked   = sorted(zip(sentences, scores), key=lambda x: -x[1])
        selected = {s for s, _ in ranked[:top_n]}
        return [s for s in sentences if s in selected]

# ══════════════════════════════════════════════════════════════
# Baseline — Lead Sentence
# ══════════════════════════════════════════════════════════════

class LeadSentenceBaseline:
    """
    Trivial baseline that always returns the first top_n sentences.

    Why this exists: In summarization research, simply taking the leading
    sentences of a document is a surprisingly strong baseline (especially
    for news articles and docstrings, where important information tends
    to come first). If the four ranking models cannot beat this baseline,
    their added complexity is not justified.

    A supervisor or reviewer will always ask: "Did you compare against
    just taking the first sentence?" This model answers that question.
    No fitting required — it is stateless.
    """

    def fit(self, corpus: list[str]) -> "LeadSentenceBaseline":
        """No-op: baseline needs no training."""
        return self

    def summarize(self, sentences: list[str], top_n: int = 2) -> list[str]:
        """Return the first top_n sentences verbatim."""
        return sentences[:top_n]


# ══════════════════════════════════════════════════════════════
# ROUGE Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_summaries(
    reference:   str,
    hypotheses:  dict[str, list[str]],
    reference_is_module_doc: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Score each model's summary against a reference string using ROUGE-1,
    ROUGE-2, and ROUGE-L F-measures (with stemming enabled).

    Args:
        reference:   The gold-standard text to compare against.
                     Ideally a hand-written summary of the file; falls back to
                     the module-level docstring when none is available.
                     NOTE: Scoring against the module docstring is a weak proxy —
                     it penalises models that surface good method-level sentences
                     not mentioned in the module docstring.  For a rigorous
                     evaluation, provide hand-written reference summaries for
                     each test file (see --reference flag in app.py).
        hypotheses:  Mapping of {model_name: [summary_sentences]}.
        reference_is_module_doc: When True, a warning is attached to each
                     model's result dict so callers can surface the caveat.

    Returns:
        Nested dict of {model_name: {rouge1, rouge2, rougeL, weak_reference}}
        float scores. ``weak_reference`` is True when the module docstring was
        used as the reference instead of a hand-written summary.

    Requires: pip install rouge-score
    """
    from rouge_score import rouge_scorer as rouge_lib

    scorer  = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results: dict[str, dict[str, float]] = {}

    for model_name, sentences in hypotheses.items():
        hypothesis = " ".join(sentences)
        scores     = scorer.score(reference, hypothesis)
        results[model_name] = {
            "rouge1":           round(scores["rouge1"].fmeasure, 4),
            "rouge2":           round(scores["rouge2"].fmeasure, 4),
            "rougeL":           round(scores["rougeL"].fmeasure, 4),
            "weak_reference":   reference_is_module_doc,
        }

    return results


def evaluate_bertscore(
    reference:  str,
    hypotheses: dict[str, list[str]],
    reference_is_module_doc: bool = True,
    lang:       str = "en",
) -> dict[str, dict[str, float]]:
    """
    Score each model's summary against a reference using BERTScore.

    BERTScore uses contextual BERT embeddings to measure semantic similarity,
    making it far less sensitive to the circular wording problem that plagues
    ROUGE when the reference is the module docstring. Two summaries that say
    the same thing differently will score highly here but poorly on ROUGE.

    Args:
        reference:   Gold-standard text (hand-written or module docstring).
        hypotheses:  Mapping of {model_name: [summary_sentences]}.
        reference_is_module_doc: Passed through so callers can display a caveat.
        lang:        Language code for BERTScore (default "en").

    Returns:
        Nested dict of {model_name: {bertscore_p, bertscore_r, bertscore_f1,
        weak_reference}} with float scores.

    Requires: pip install bert-score
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "BERTScore is not installed. Run: pip install bert-score\n"
            "Then re-run with --metric bertscore or --all-metrics."
        )

    model_names = list(hypotheses.keys())
    hypotheses_text = [" ".join(hypotheses[m]) for m in model_names]
    references_text = [reference] * len(model_names)

    # bert_score returns tensors of shape (N,); convert to plain floats
    P, R, F1 = bert_score_fn(
        hypotheses_text,
        references_text,
        lang=lang,
        verbose=False,
    )

    results: dict[str, dict[str, float]] = {}
    for i, model_name in enumerate(model_names):
        results[model_name] = {
            "bertscore_p":      round(float(P[i]), 4),
            "bertscore_r":      round(float(R[i]), 4),
            "bertscore_f1":     round(float(F1[i]), 4),
            "weak_reference":   reference_is_module_doc,
        }

    return results


def evaluate_coverage(
    candidates: list[str],
    hypotheses: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """
    Reference-free evaluation: measure how well each summary covers the key
    concepts in the full candidate pool.

    Coverage is defined as the fraction of unique content tokens in the full
    candidate pool that appear in the summary. A high coverage score means the
    model surfaced sentences that collectively mention most of the file's
    important concepts.

    Diversity is defined as 1 minus the ratio of repeated tokens to total
    tokens in the summary. A high diversity score means the model avoided
    repeating the same idea across its selected sentences.

    Both scores are reference-free — no module docstring or hand-written
    summary is needed. This makes them immune to the circular scoring problem
    that affects ROUGE and BERTScore when the reference is the module docstring.

    Args:
        candidates: Full flat list of docstring sentences from the target file.
        hypotheses: Mapping of {model_name: [summary_sentences]}.

    Returns:
        Nested dict of {model_name: {coverage, diversity, combined}}
        where combined = 0.5 * coverage + 0.5 * diversity.
    """
    # Build the full concept vocabulary from all candidates
    full_tokens: set[str] = set()
    for sent in candidates:
        full_tokens.update(tokenize(sent))

    if not full_tokens:
        return {m: {"coverage": 0.0, "diversity": 0.0, "combined": 0.0}
                for m in hypotheses}

    results: dict[str, dict[str, float]] = {}
    for model_name, sentences in hypotheses.items():
        if not sentences:
            results[model_name] = {"coverage": 0.0, "diversity": 0.0, "combined": 0.0}
            continue

        summary_tokens_all: list[str] = []
        for sent in sentences:
            summary_tokens_all.extend(tokenize(sent))

        # Coverage: fraction of full-file concepts mentioned in the summary
        summary_token_set  = set(summary_tokens_all)
        coverage           = len(summary_token_set & full_tokens) / len(full_tokens)

        # Diversity: penalise repetition within the summary itself
        total = len(summary_tokens_all)
        if total > 0:
            from collections import Counter
            token_counts = Counter(summary_tokens_all)
            repeated     = sum(c - 1 for c in token_counts.values() if c > 1)
            diversity    = 1.0 - (repeated / total)
        else:
            diversity = 0.0

        combined = round(0.5 * coverage + 0.5 * diversity, 4)
        results[model_name] = {
            "coverage":  round(coverage, 4),
            "diversity": round(diversity, 4),
            "combined":  combined,
        }

    return results


def select_best_model(
    scores:  dict[str, dict[str, float]],
    metric:  str = "rougeL",
) -> str:
    """
    Return the name of the model with the highest score on the given metric.

    Works across all three evaluation dicts (ROUGE, BERTScore, Coverage).
    Supported metric keys:
      ROUGE:      rouge1, rouge2, rougeL
      BERTScore:  bertscore_p, bertscore_r, bertscore_f1
      Coverage:   coverage, diversity, combined

    Defaults to rougeL as it rewards longest common subsequence matches and is
    the most balanced single metric for extractive summarization quality.
    """
    return max(scores, key=lambda m: scores[m].get(metric, 0.0))


# ══════════════════════════════════════════════════════════════
# Usage Example Generation
# ══════════════════════════════════════════════════════════════

def generate_usage_example(metadata: dict, filename: str) -> str:
    """
    Auto-generate a minimal Quick Start code block from the parsed metadata.
    Detects the first public class with an __init__ method and emits an
    instantiation call followed by the first two non-init public method calls.
    Falls back to global functions when no classes are present.
    """
    module = filename.replace(".py", "")
    lines  = [f"## Quick start\n\n```python", f"import {module}", ""]

    if metadata["classes"]:
        cls  = metadata["classes"][0]
        init = next((m for m in cls["methods"] if m["name"] == "__init__"), None)
        init_args = (
            ", ".join(f"<{a}>" for a in init["args"]) if init and init["args"] else ""
        )
        lines.append(f"obj = {cls['name']}({init_args})")

        public_methods = [m for m in cls["methods"] if m["name"] != "__init__"]
        for m in public_methods[:2]:
            m_args = ", ".join(f"<{a}>" for a in m["args"])
            lines.append(f"result = obj.{m['name']}({m_args})")

    elif metadata["functions"]:
        for fn in metadata["functions"][:2]:
            fn_args = ", ".join(f"<{a}>" for a in fn["args"])
            lines.append(f"result = {fn['name']}({fn_args})")

    lines.append("```\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Markdown Generation
# ══════════════════════════════════════════════════════════════

def generate_markdown(
    metadata:          dict,
    filename:          str,
    summary_sentences: list[str],
    rouge_scores:      dict[str, dict[str, float]] | None = None,
    bertscore_scores:  dict[str, dict[str, float]] | None = None,
    coverage_scores:   dict[str, dict[str, float]] | None = None,
    winning_model:     str | None = None,
) -> str:
    """
    Render structured, nested Markdown documentation from parsed AST metadata.

    Sections produced (in order):
      1. Title
      2. Description  — elected summary sentences
      3. Model Evaluation — ROUGE, BERTScore, and Coverage tables (optional)
      4. Quick Start  — auto-generated usage example
      5. Dependencies — top-level imports
      6. Classes      — with docstrings and per-method signatures + descriptions
      7. Global Functions — with docstrings

    Args:
        metadata:          Output of parse_python_file().
        filename:          Basename of the source file (e.g. "example.py").
        summary_sentences: Sentences selected by the winning model.
        rouge_scores:      Optional dict from evaluate_summaries().
        bertscore_scores:  Optional dict from evaluate_bertscore().
        coverage_scores:   Optional dict from evaluate_coverage().
        winning_model:     Name of the model that produced the summary.
    """
    title = filename.replace(".py", "").replace("_", " ").title()
    lines = [f"# {title}\n"]

    # ── 1. Description ───────────────────────────────────────
    lines.append("## Description\n")
    if summary_sentences:
        for s in summary_sentences:
            lines.append(f"> {s}")
    else:
        lines.append("> *No overview description could be generated.*")
    lines.append("")

    if winning_model:
        lines.append(f"*Summary generated by: **{winning_model.upper()}***\n")

    # ── 2a. ROUGE Evaluation Table ───────────────────────────
    if rouge_scores:
        lines.append("## Model evaluation (ROUGE)\n")
        lines.append("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |")
        lines.append("|-------|---------|---------|---------|")
        for model, scores in sorted(rouge_scores.items()):
            marker = " ✓" if model == winning_model else ""
            lines.append(
                f"| {model}{marker} "
                f"| {scores['rouge1']:.4f} "
                f"| {scores['rouge2']:.4f} "
                f"| {scores['rougeL']:.4f} |"
            )
        lines.append("")
        any_weak = any(v.get("weak_reference", False) for v in rouge_scores.values())
        if any_weak:
            lines.append(
                "> ⚠ Reference = module docstring (weak proxy). "
                "Scores may be unfairly low for models that surface good "
                "method-level sentences not in the module doc. "
                "Use `--reference <file>` for a hand-written summary.\n"
            )

    # ── 2b. BERTScore Evaluation Table ───────────────────────
    if bertscore_scores:
        lines.append("## Model evaluation (BERTScore)\n")
        lines.append("| Model | Precision | Recall | F1 |")
        lines.append("|-------|-----------|--------|----|")
        for model, scores in sorted(bertscore_scores.items()):
            marker = " ✓" if model == winning_model else ""
            lines.append(
                f"| {model}{marker} "
                f"| {scores['bertscore_p']:.4f} "
                f"| {scores['bertscore_r']:.4f} "
                f"| {scores['bertscore_f1']:.4f} |"
            )
        lines.append("")

    # ── 2c. Coverage Evaluation Table (reference-free) ───────
    if coverage_scores:
        lines.append("## Model evaluation (Coverage — reference-free)\n")
        lines.append("| Model | Coverage | Diversity | Combined |")
        lines.append("|-------|----------|-----------|----------|")
        for model, scores in sorted(coverage_scores.items()):
            marker = " ✓" if model == winning_model else ""
            lines.append(
                f"| {model}{marker} "
                f"| {scores['coverage']:.4f} "
                f"| {scores['diversity']:.4f} "
                f"| {scores['combined']:.4f} |"
            )
        lines.append("")

    # ── 3. Quick Start ────────────────────────────────────────
    lines.append(generate_usage_example(metadata, filename))

    # ── 5. Dependencies ───────────────────────────────────────
    if metadata["imports"]:
        lines.append("## Dependencies\n")
        lines.append("```python")
        for imp in metadata["imports"]:
            lines.append(imp.strip())
        lines.append("```\n")

    # ── 5. Classes ────────────────────────────────────────────
    if metadata["classes"]:
        lines.append("## Classes\n")
        for cls in metadata["classes"]:
            lines.append(f"### `class {cls['name']}`\n")
            if cls["docstring"]:
                lines.append(f"{cls['docstring']}\n")

            public_methods = [m for m in cls["methods"] if m["name"] != "__init__"]
            if public_methods:
                lines.append("#### Methods\n")
                for method in public_methods:
                    sig = f"`{method['name']}({', '.join(method['args'])})`"
                    lines.append(f"**{sig}**")
                    if method["docstring"]:
                        # Indent multi-line docstrings cleanly under the signature
                        for doc_line in method["docstring"].splitlines():
                            lines.append(f"{doc_line.strip()}")
                    lines.append("")

    # ── 6. Global Functions ───────────────────────────────────
    if metadata["functions"]:
        lines.append("## Global functions\n")
        for fn in metadata["functions"]:
            lines.append(f"### `{fn['name']}({', '.join(fn['args'])})`\n")
            if fn["docstring"]:
                lines.append(f"{fn['docstring']}\n")

    return "\n".join(lines)