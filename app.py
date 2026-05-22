# app.py
"""
Entry point for the local README generator pipeline.

Usage:
    python app.py <path_to_python_file.py> [--output <output_path.md>] [--metric <metric>]

Examples:
    python app.py src/my_module.py
    python app.py src/my_module.py --output docs/README.md
    python app.py src/my_module.py --metric bertscore_f1
    python app.py src/my_module.py --all-metrics
    python app.py src/my_module.py --batch ./test_corpus
    python app.py src/my_module.py --batch ./test_corpus --reference ./references

How it works:
    1. Scans the target file's project directory to build a local training corpus
       from all .py docstrings — no internet connection or pre-trained models needed.
    2. Fits three comparison models (TF-IDF, BM25, LSA) and a Lead Sentence baseline on that corpus.
    3. Parses the target file with Python's AST to extract structured metadata
       and a flat candidate pool of docstring sentences.
    4. Runs all four models and evaluates each summary with three metrics:
         ROUGE     — n-gram overlap (rouge1, rouge2, rougeL)
         BERTScore — semantic similarity via BERT embeddings (bertscore_f1)
         Coverage  — reference-free concept coverage (combined)
    5. The model with the highest score on the chosen metric wins and its
       summary is used in the final README.

Dependencies:
    pip install numpy rank-bm25 rouge-score bert-score
"""

import sys
import os
import glob
import argparse
import numpy as np

import readme_engine as re_eng


# ══════════════════════════════════════════════════════════════
# Local Corpus Builder
# ══════════════════════════════════════════════════════════════

def build_local_corpus(root_dir: str, verbose: bool = True) -> list[str]:
    """
    Recursively scan all .py files under root_dir and collect every docstring
    sentence found by the AST parser into a flat corpus list.

    This replaces any external dataset download — the project's own source code
    serves as the training distribution, which is domain-appropriate and always
    available offline.

    Args:
        root_dir: Directory to scan (usually the target file's parent directory).
        verbose:  Print progress information when True.

    Returns:
        A list of docstring sentences usable as a training corpus.
        Falls back to a single placeholder string if no docstrings are found,
        so downstream model .fit() calls never receive an empty list.
    """
    corpus: list[str] = []
    py_files = glob.glob(os.path.join(root_dir, "**/*.py"), recursive=True)

    if verbose:
        print(f"[Corpus] Scanning {len(py_files)} Python file(s) in '{root_dir}'...")

    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            meta = re_eng.parse_python_file(source)
            corpus.extend(meta["docstrings"])
        except Exception as exc:
            if verbose:
                print(f"[Corpus] Skipping '{path}': {exc}")

    if not corpus:
        if verbose:
            print("[Corpus] No docstrings found — using placeholder corpus.")
        return ["no documentation available for this project"]

    if verbose:
        print(f"[Corpus] Collected {len(corpus)} sentence(s) from {len(py_files)} file(s).")

    return corpus


# ══════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════

def fit_all_models(
    corpus: list[str],
    verbose: bool = True,
) -> tuple[re_eng.TFIDFModel, re_eng.BM25Model, re_eng.LSAModel, re_eng.LeadSentenceBaseline]:
    """
    Fit the three comparison models (TF-IDF, BM25, LSA) on the local corpus
    and initialise the Lead Sentence baseline. Return all four as a tuple.

    The baseline is stateless (no training needed) but is included here so
    the caller always receives everything it needs from a single call.

    Args:
        corpus:  List of training sentences (output of build_local_corpus).
        verbose: Print status messages when True.

    Returns:
        Tuple of (TFIDFModel, BM25Model, LSAModel, LeadSentenceBaseline),
        all ready to call .summarize() on.
    """
    if verbose:
        print("[Models] Fitting TF-IDF...")
    tfidf_model = re_eng.TFIDFModel().fit(corpus)

    if verbose:
        print("[Models] Fitting BM25...")
    bm25_model = re_eng.BM25Model().fit(corpus)

    if verbose:
        print("[Models] Fitting LSA...")
    lsa_model = re_eng.LSAModel().fit(corpus)

    if verbose:
        print("[Models] Initialising Lead Sentence baseline (no training required)...")
    baseline_model = re_eng.LeadSentenceBaseline()

    return tfidf_model, bm25_model, lsa_model, baseline_model


# ══════════════════════════════════════════════════════════════
# Summary Generation & Evaluation
# ══════════════════════════════════════════════════════════════

def run_all_summaries(
    candidates:      list[str],
    tfidf_model:     re_eng.TFIDFModel,
    bm25_model:      re_eng.BM25Model,
    lsa_model:       re_eng.LSAModel,
    baseline_model:  re_eng.LeadSentenceBaseline,
    top_n:           int = 2,
) -> dict[str, list[str]]:
    """
    Run the three comparison models and the Lead Sentence baseline on the
    candidate sentences and return all four summaries.

    The baseline is always run so the ROUGE table immediately shows whether
    any of the three models outperform simply taking the first sentence —
    the minimum bar every summarizer must clear.

    Args:
        candidates:  Flat list of docstring sentences from the target file.
        top_n:       Number of sentences each model should select.

    Returns:
        Dict mapping model name → list of selected summary sentences.
        Keys: "lead" (baseline), "tfidf", "bm25", "lsa".
    """
    return {
        "lead":  baseline_model.summarize(candidates, top_n=top_n),
        "tfidf": tfidf_model.summarize(candidates,    top_n=top_n),
        "bm25":  bm25_model.summarize(candidates,     top_n=top_n),
        "lsa":   lsa_model.summarize(candidates,      top_n=top_n),
    }


# ══════════════════════════════════════════════════════════════
# Evaluation Table Printer
# FIX #10: moved to module level so it is independently testable.
# ══════════════════════════════════════════════════════════════

def print_eval_table(
    rouge_scores:     dict[str, dict[str, float]],
    bertscore_scores: dict[str, dict[str, float]] | None,
    coverage_scores:  dict[str, dict[str, float]] | None,
    winning_model:    str,
    winning_metric:   str,
) -> None:
    """Pretty-print all three evaluation tables to stdout."""

    # ── ROUGE ────────────────────────────────────────────────
    any_weak = any(v.get("weak_reference", False) for v in rouge_scores.values())
    print("\n" + "─" * 52)
    print(f"  {'Model':<14} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
    print("─" * 52)
    for model, scores in sorted(rouge_scores.items()):
        marker = " ✓" if (model == winning_model and winning_metric.startswith("rouge")) else "  "
        print(
            f"  {model + marker:<14}"
            f" {scores['rouge1']:>8.4f}"
            f" {scores['rouge2']:>8.4f}"
            f" {scores['rougeL']:>8.4f}"
        )
    print("─" * 52)

    if any_weak:
        print(
            "  ⚠  Reference = module docstring (weak proxy).\n"
            "     Use --reference <dir_or_file> to supply hand-written summary documentation.\n"
        )

    # ── BERTScore ────────────────────────────────────────────
    if bertscore_scores:
        print("─" * 52)
        print(f"  {'Model':<14} {'BS-Prec':>8} {'BS-Rec':>8} {'BS-F1':>8}")
        print("─" * 52)
        for model, scores in sorted(bertscore_scores.items()):
            marker = " ✓" if (model == winning_model and winning_metric.startswith("bertscore")) else "  "
            print(
                f"  {model + marker:<14}"
                f" {scores['bertscore_p']:>8.4f}"
                f" {scores['bertscore_r']:>8.4f}"
                f" {scores['bertscore_f1']:>8.4f}"
            )
        print("─" * 52 + "\n")

    # ── Coverage (reference-free) ────────────────────────────
    if coverage_scores:
        print("─" * 52)
        print(f"  {'Model':<14} {'Coverage':>9} {'Diversity':>9} {'Combined':>9}")
        print("─" * 52)
        for model, scores in sorted(coverage_scores.items()):
            marker = " ✓" if (model == winning_model and winning_metric in ("coverage", "diversity", "combined")) else "  "
            print(
                f"  {model + marker:<14}"
                f" {scores['coverage']:>9.4f}"
                f" {scores['diversity']:>9.4f}"
                f" {scores['combined']:>9.4f}"
            )
        print("─" * 52 + "\n")

    print(f"  Winner ({winning_metric}): {winning_model.upper()}\n")


def _print_batch_summary(label: str, data: dict[str, list[float]], n_files: int) -> None:
    """
    Pretty-print the batch summary table for one metric group.

    FIX #10: extracted from inside main() so it is independently importable
    and testable without invoking the full CLI.

    Args:
        label:   Metric label shown in the table header (e.g. "rougeL").
        data:    Mapping of model name → list of per-file scores.
        n_files: Total number of files processed (used in the header line).
    """
    if not data:
        return
    print(f"\n{'─' * 56}")
    print(f"  Batch summary — {label} across {n_files} file(s)")
    print(f"{'─' * 56}")
    print(f"  {'Model':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'─' * 56}")
    for model, vals in sorted(data.items()):
        arr = np.array(vals)
        print(
            f"  {model:<14}"
            f" {arr.mean():>8.4f}"
            f" {arr.std():>8.4f}"
            f" {arr.min():>8.4f}"
            f" {arr.max():>8.4f}"
        )
    print(f"{'─' * 56}")


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a README for a Python file using local NLP models."
    )
    parser.add_argument(
        "filepath",
        help="Path to the Python source file to document.",
    )
    parser.add_argument(
        "--output", "-o",
        default="README.md",
        help="Output path for the generated Markdown file (default: README.md).",
    )
    parser.add_argument(
        "--metric", "-m",
        default="rougeL",
        choices=[
            "rouge1", "rouge2", "rougeL",
            "bertscore_p", "bertscore_r", "bertscore_f1",
            "coverage", "diversity", "combined",
        ],
        help=(
            "Metric used to elect the winning model (default: rougeL). "
            "ROUGE options: rouge1, rouge2, rougeL. "
            "BERTScore options: bertscore_p, bertscore_r, bertscore_f1. "
            "Reference-free options: coverage, diversity, combined."
        ),
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help=(
            "Run and display all three evaluation metrics (ROUGE, BERTScore, Coverage). "
            "The winning model is still elected by --metric. "
            "BERTScore requires: pip install bert-score"
        ),
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=2,
        help="Number of sentences each model selects for its summary (default: 2).",
    )
    parser.add_argument(
        "--reference", "-r",
        default=None,
        help=(
            "Path to a file or directory containing hand-written reference summaries. "
            "In single-file mode, specify a plain text file path. In batch mode, "
            "specify a directory containing reference text files whose filenames match "
            "the source file stems (e.g., 'scaler.txt' matches 'scaler.py')."
        ),
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        help=(
            "Path to a directory. When set, the pipeline runs on every .py file "
            "in that directory, prints per-file ROUGE scores, and reports mean ± "
            "std across all files — giving a statistically meaningful comparison. "
            "At least 10–15 files are recommended."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser.parse_args()


def run_single_file(
    target_filepath: str,
    args: argparse.Namespace,
    tfidf_model: re_eng.TFIDFModel,
    bm25_model: re_eng.BM25Model,
    lsa_model: re_eng.LSAModel,
    baseline_model: re_eng.LeadSentenceBaseline,
    verbose: bool = True,
    custom_reference: str | None = None,
) -> tuple:
    """
    Run the full summarization + evaluation pipeline on one file.

    Always computes ROUGE and Coverage. BERTScore is computed when
    args.all_metrics is True or when args.metric starts with 'bertscore'.

    FIX #3: args.metric is never mutated here. When BERTScore import fails and
    the user requested a bertscore metric, the fallback metric is stored in the
    local variable ``effective_metric`` rather than overwriting args.metric.
    This prevents batch mode from silently switching all subsequent files to
    rougeL after the first import failure.

    Args:
        custom_reference: Pre-loaded hand-written reference text, or None to
                          fall back to the module docstring.

    Returns:
        (rouge_scores, bertscore_scores, coverage_scores,
         winning_model, hypotheses, metadata)
    """
    if verbose:
        print(f"[Parse] Processing: {target_filepath}")
    with open(target_filepath, "r", encoding="utf-8") as f:
        source_code = f.read()

    metadata   = re_eng.parse_python_file(source_code)
    candidates = metadata["docstrings"]
    if len(candidates) < 2:
        candidates = re_eng.split_sentences(metadata["description"])
    if not candidates:
        candidates = ["No docstrings found in this file."]

    if verbose:
        print(f"[Summarize] Running {args.top_n}-sentence summaries on "
              f"{len(candidates)} candidate(s)...")

    hypotheses = run_all_summaries(
        candidates, tfidf_model, bm25_model, lsa_model,
        baseline_model, top_n=args.top_n,
    )

    # Determine reference and whether it is a weak proxy
    if custom_reference:
        reference         = custom_reference
        reference_is_weak = False
    else:
        reference         = metadata["description"] or candidates[0]
        reference_is_weak = True

    # FIX #3: use a local copy of the metric so we never mutate args.metric.
    effective_metric = args.metric

    # ── ROUGE (always) ───────────────────────────────────────
    if verbose:
        print(f"[ROUGE] Evaluating summaries — "
              f"{'hand-written' if not reference_is_weak else 'module-doc'} reference...")
    rouge_scores = re_eng.evaluate_summaries(
        reference, hypotheses, reference_is_module_doc=reference_is_weak
    )

    # ── BERTScore (when requested) ───────────────────────────
    want_bertscore = getattr(args, "all_metrics", False) or effective_metric.startswith("bertscore")
    bertscore_scores: dict | None = None
    if want_bertscore:
        if verbose:
            print("[BERTScore] Computing semantic similarity scores...")
        try:
            bertscore_scores = re_eng.evaluate_bertscore(
                reference, hypotheses, reference_is_module_doc=reference_is_weak
            )
        except ImportError as e:
            print(f"  ⚠  {e}")
            if effective_metric.startswith("bertscore"):
                print("  Falling back to rougeL for model election.")
                # FIX #3: write to local variable only — args.metric is untouched.
                effective_metric = "rougeL"

    # ── Coverage (always — reference-free) ───────────────────
    if verbose:
        print("[Coverage] Computing reference-free coverage scores...")
    coverage_scores = re_eng.evaluate_coverage(candidates, hypotheses)

    # ── Elect winner ─────────────────────────────────────────
    if effective_metric.startswith("bertscore") and bertscore_scores:
        winning_model = re_eng.select_best_model(bertscore_scores, metric=effective_metric)
    elif effective_metric in ("coverage", "diversity", "combined"):
        winning_model = re_eng.select_best_model(coverage_scores, metric=effective_metric)
    else:
        winning_model = re_eng.select_best_model(rouge_scores, metric=effective_metric)

    if verbose:
        print_eval_table(rouge_scores, bertscore_scores, coverage_scores,
                         winning_model, effective_metric)

    return rouge_scores, bertscore_scores, coverage_scores, winning_model, hypotheses, metadata


def main() -> None:
    args    = parse_args()
    verbose = not args.quiet

    # ── Validate input ────────────────────────────────────────
    target_filepath = os.path.abspath(args.filepath)
    if not os.path.exists(target_filepath):
        print(f"[Error] File not found: '{target_filepath}'")
        sys.exit(1)
    if not target_filepath.endswith(".py"):
        print(f"[Error] Expected a .py file, got: '{target_filepath}'")
        sys.exit(1)

    # ── Handle reference parameter ────────────────────────────
    custom_reference = None
    reference_is_dir = False

    if args.reference:
        ref_path = os.path.abspath(args.reference)
        if not os.path.exists(ref_path):
            print(f"[Error] Reference path not found: '{ref_path}'")
            sys.exit(1)

        if os.path.isdir(ref_path):
            reference_is_dir = True
            if verbose:
                print(f"[Reference] Set reference folder to '{ref_path}' for automated dynamic matching.")
        else:
            with open(ref_path, "r", encoding="utf-8") as f:
                custom_reference = f.read().strip()
            if verbose:
                print(f"[Reference] Loaded single hand-written reference from '{ref_path}'")
    else:
        if verbose:
            print(
                "[Reference] No --reference path provided. ROUGE will be scored "
                "against the module docstring (weak proxy — see output for caveat)."
            )

    # ── Step 1: Build local corpus ────────────────────────────
    project_dir = os.path.dirname(target_filepath)
    corpus      = build_local_corpus(project_dir, verbose=verbose)

    # ── Step 2: Fit all models (including baseline) ───────────
    tfidf_model, bm25_model, lsa_model, baseline_model = fit_all_models(
        corpus, verbose=verbose
    )

    # ── Batch mode: run on every .py file in --batch dir ─────
    if args.batch:
        batch_dir  = os.path.abspath(args.batch)
        py_files   = glob.glob(os.path.join(batch_dir, "**/*.py"), recursive=True)
        if not py_files:
            print(f"[Batch] No .py files found in '{batch_dir}'")
            sys.exit(1)

        print(f"\n[Batch] Running on {len(py_files)} file(s) in '{batch_dir}'...")

        # Accumulate per-model scores across files for each metric group
        all_rouge:    dict[str, list[float]] = {}
        all_bert:     dict[str, list[float]] = {}
        all_coverage: dict[str, list[float]] = {}

        for py_path in py_files:
            print(f"\n  ── {os.path.basename(py_path)} ──")

            # Resolve the reference dynamically for this specific file iteration
            file_specific_reference = None
            if args.reference:
                if reference_is_dir:
                    stem = os.path.splitext(os.path.basename(py_path))[0]
                    potential_ref_path = os.path.join(os.path.abspath(args.reference), f"{stem}.txt")
                    if os.path.exists(potential_ref_path):
                        with open(potential_ref_path, "r", encoding="utf-8") as f:
                            file_specific_reference = f.read().strip()
                    else:
                        if not args.quiet:
                            print(f"    [Warning] Reference text profile '{stem}.txt' missing in directory. Using weak proxy.")
                else:
                    # Fall back to the single loaded reference if directory logic was not targeted
                    file_specific_reference = custom_reference

            try:
                rouge_s, bert_s, cov_s, _, _, _ = run_single_file(
                    py_path, args,
                    tfidf_model, bm25_model, lsa_model, baseline_model,
                    verbose=False,
                    custom_reference=file_specific_reference,
                )
                for model, s in rouge_s.items():
                    all_rouge.setdefault(model, []).append(s["rougeL"])
                    print(f"    {model:<12} rougeL: {s['rougeL']:.4f}", end="")
                    if bert_s and model in bert_s:
                        all_bert.setdefault(model, []).append(bert_s[model]["bertscore_f1"])
                        print(f"   bertscore_f1: {bert_s[model]['bertscore_f1']:.4f}", end="")
                    if cov_s and model in cov_s:
                        all_coverage.setdefault(model, []).append(cov_s[model]["combined"])
                        print(f"   coverage: {cov_s[model]['combined']:.4f}", end="")
                    print()
            except Exception as exc:
                print(f"    [Skip] {exc}")

        n = len(py_files)
        _print_batch_summary("rougeL",       all_rouge,    n)
        _print_batch_summary("bertscore_f1", all_bert,     n)
        _print_batch_summary("coverage",     all_coverage, n)

        # Warn if sample is too small for statistical reliability
        if n < 10:
            print(
                f"\n  ⚠  Only {n} file(s) tested. "
                "Results have high variance.\n"
                "     Use 10–15 files minimum for a reliable comparison.\n"
            )
        return

    # ── Single-file mode ──────────────────────────────────────
    rouge_scores, bertscore_scores, coverage_scores, winning_model, hypotheses, metadata = run_single_file(
        target_filepath, args,
        tfidf_model, bm25_model, lsa_model, baseline_model,
        verbose=verbose,
        custom_reference=custom_reference,
    )

    champion_summary = hypotheses[winning_model]

    # ── Generate Markdown ─────────────────────────────────────
    if verbose:
        print(f"[Markdown] Generating README with {winning_model.upper()} summary...")

    final_readme = re_eng.generate_markdown(
        metadata          = metadata,
        filename          = os.path.basename(target_filepath),
        summary_sentences = champion_summary,
        rouge_scores      = rouge_scores,
        bertscore_scores  = bertscore_scores,
        coverage_scores   = coverage_scores,
        winning_model     = winning_model,
    )

    # ── Write output ──────────────────────────────────────────
    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_readme)

    print(f"[Done] README written to '{output_path}'")


if __name__ == "__main__":
    main()