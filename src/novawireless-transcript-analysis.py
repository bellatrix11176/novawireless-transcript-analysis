#!/usr/bin/env python3
"""
novawireless_nlp_trust_signals.py

NLP TRUST SIGNALS PIPELINE (Lab-Aligned Rewrite)

Linguistic analysis of NovaWireless Call Center Lab transcripts.
Identifies which words and phrases are distinctive markers of
Goodhart divergence, gaming behavior, bandaid credit usage, and
repeat contact failure.

What it does (end-to-end):

1) Loads data/calls_sanitized_*.csv (monthly files from Call Center Lab)
2) Extracts and cleans transcript_text column
3) Builds TF-IDF term-document matrix
4) Runs term lift analysis against FOUR outcome targets:
   a) proxy_vs_true_gap — words that distinguish inflated proxy calls
   b) scenario_category — words that distinguish clean/gaming/fraud
   c) bandaid_credit — words that mark unauthorized suppression credits
   d) repeat_contact_30d — words that predict downstream repeat calls
5) Scans transcripts for profanity patterns, measures rates by outcome
6) Generates charts and a written summary report

Data requirement:
  data/calls_sanitized_2025-01.csv through calls_sanitized_2025-12.csv
  (or any subset — pipeline adapts)

Usage:
  python src/novawireless_nlp_trust_signals.py
  python src/novawireless_nlp_trust_signals.py --max_features 30000
  python src/novawireless_nlp_trust_signals.py --top_n 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS


# ============================================================
# Repo-root helpers
# ============================================================

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(50):
        if (cur / "data").is_dir() and (cur / "src").is_dir():
            (cur / "output").mkdir(parents=True, exist_ok=True)
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find repo root containing data/ and src/.\n"
        "Run this script from somewhere inside your repo."
    )


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ============================================================
# Constants
# ============================================================

# Columns needed from calls_sanitized
REQUIRED_COLUMNS = [
    "call_id", "transcript_text", "scenario",
    "true_resolution", "resolution_flag",
    "credit_type", "credit_applied",
]

OPTIONAL_COLUMNS = [
    "repeat_contact_30d", "credit_authorized",
    "rep_gaming_propensity", "rep_burnout_level",
    "escalation_flag", "call_type",
]

# Scenario category mapping
CLEAN_SCENARIOS = {"clean", "activation_clean", "line_add_legitimate"}
GAMING_SCENARIOS = {"gamed_metric"}
FRAUD_SCENARIOS = {
    "fraud_store_promo", "fraud_line_add",
    "fraud_hic_exchange", "fraud_care_promo",
}
AMBIGUOUS_SCENARIOS = {"unresolvable_clean", "activation_failed"}

# Stopwords: sklearn defaults + NovaWireless template artifacts
CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    # Template artifacts from transcript builder
    "novalink", "novawireless", "wireless", "redefining",
    "rep", "customer", "agent",
    "thank", "thanks", "calling", "today", "together",
    "okay", "confirmed", "checking", "verifying",
    "ll", "ve", "don", "doesn", "isn", "im", "youre", "well",
    "end", "hi", "hello", "help", "sure", "know", "want",
    "let", "going", "see", "look", "just", "like", "right",
    "got", "yes", "yeah", "no", "oh",
})

# Profanity lexicon
PROFANITY_PATTERNS = [
    re.compile(r"\bfuck\w*\b", re.I),
    re.compile(r"\bshit\w*\b", re.I),
    re.compile(r"\bbullshit\b", re.I),
    re.compile(r"\bbitch\w*\b", re.I),
    re.compile(r"\basshole\w*\b", re.I),
    re.compile(r"\bdamn\w*\b", re.I),
    re.compile(r"\bgoddamn\w*\b", re.I),
    re.compile(r"\bhell\b", re.I),
    re.compile(r"\bwtf\b", re.I),
    re.compile(r"\bcrap\w*\b", re.I),
    re.compile(r"\bpiss\w*\b", re.I),
]

# PII redaction patterns (synthetic data, but good practice)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){1}\d{3}[-.\s]?\d{4}\b")
RE_LONG_DIGITS = re.compile(r"\b\d{9,}\b")
RE_DOLLAR = re.compile(r"\$\d+\.?\d*")

# Chart styling
CHART_DPI = 180
CHART_BG = "#1a1a2e"
CHART_FG = "#e0e0e0"
CHART_ACCENT = "#00d4aa"
CHART_WARN = "#ff6b6b"


# ============================================================
# Data loading
# ============================================================

def load_monthly_files(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all calls_sanitized_*.csv files."""
    pattern = "calls_sanitized_*.csv"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}\n"
            "Place your calls_sanitized_YYYY-MM.csv files in data/ and rerun."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(files)} files\n")
    return combined


# ============================================================
# Text cleaning
# ============================================================

def _coerce_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "t", "yes", "y"}).astype(int)


def clean_text(text: str, max_chars: int = 5000) -> str:
    """Clean a single transcript: redact PII, normalize whitespace, truncate."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text
    t = RE_EMAIL.sub("[EMAIL]", t)
    t = RE_PHONE.sub("[PHONE]", t)
    t = RE_LONG_DIGITS.sub("[ID]", t)
    t = RE_DOLLAR.sub("[AMT]", t)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Truncate
    if len(t) > max_chars:
        t = t[:max_chars]
    return t


def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Build analysis-ready dataframe with clean text and derived targets."""
    out = df.copy()

    # Clean transcript text
    print("  Cleaning transcripts...")
    out["clean_text"] = out["transcript_text"].fillna("").apply(clean_text)

    # Drop empty transcripts
    has_text = out["clean_text"].str.len() > 10
    dropped = (~has_text).sum()
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with empty/short transcripts")
    out = out[has_text].copy().reset_index(drop=True)

    # --- Derived targets ---

    # Target 1: proxy_vs_true_gap (binary)
    proxy = _coerce_flag(out["resolution_flag"])
    true = _coerce_flag(out["true_resolution"])
    out["target_proxy_gap"] = ((proxy == 1) & (true == 0)).astype(int)

    # Target 2: scenario_category (clean / gaming / fraud / ambiguous)
    def categorize_scenario(s):
        if s in CLEAN_SCENARIOS:
            return "clean"
        elif s in GAMING_SCENARIOS:
            return "gaming"
        elif s in FRAUD_SCENARIOS:
            return "fraud"
        else:
            return "ambiguous"
    out["target_scenario_cat"] = out["scenario"].apply(categorize_scenario)

    # Target 3: bandaid_credit (binary)
    out["target_bandaid"] = (
        out["credit_type"].astype(str).str.strip() == "bandaid"
    ).astype(int)

    # Target 4: repeat_contact_30d (binary)
    if "repeat_contact_30d" in out.columns:
        out["target_repeat_30d"] = _coerce_flag(out["repeat_contact_30d"])
    else:
        out["target_repeat_30d"] = 0

    print(f"  Corpus ready: {len(out):,} documents")
    return out


# ============================================================
# TF-IDF vectorization
# ============================================================

def build_tfidf(
    texts: pd.Series,
    max_features: int = 25000,
    min_df: int = 5,
    max_df: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Fit TF-IDF vectorizer and return (vectorizer, matrix)."""
    print(f"  Fitting TF-IDF (max_features={max_features}, ngrams={ngram_range})...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        stop_words=sorted(CUSTOM_STOPWORDS),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"  Vocabulary size: {len(vocab):,} terms")
    print(f"  Matrix shape: {matrix.shape}")
    return vectorizer, matrix.toarray()


# ============================================================
# Term lift analysis
# ============================================================

def compute_term_lift_binary(
    tfidf_matrix: np.ndarray,
    vocab: np.ndarray,
    labels: np.ndarray,
    top_n: int = 30,
    min_mean: float = 1e-6,
) -> pd.DataFrame:
    """
    For a binary target (0/1), compute term lift = pos_mean / neg_mean.
    Returns top terms by lift with statistics.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() < 10 or neg_mask.sum() < 10:
        return pd.DataFrame()

    pos_mean = tfidf_matrix[pos_mask].mean(axis=0)
    neg_mean = tfidf_matrix[neg_mask].mean(axis=0)

    # Filter near-zero means
    valid = (pos_mean > min_mean) | (neg_mean > min_mean)

    lift = np.where(
        (neg_mean > min_mean) & valid,
        pos_mean / np.maximum(neg_mean, min_mean),
        0.0,
    )

    results = pd.DataFrame({
        "term": vocab,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "lift": lift,
        "log_lift": np.log2(np.maximum(lift, 1e-10)),
        "diff": pos_mean - neg_mean,
    })

    # Filter and sort
    results = results[results["lift"] > 1.0].copy()
    results = results.sort_values("lift", ascending=False).head(top_n)
    return results.reset_index(drop=True)


def compute_term_lift_multiclass(
    tfidf_matrix: np.ndarray,
    vocab: np.ndarray,
    labels: np.ndarray,
    classes: List[str],
    top_n: int = 20,
    min_mean: float = 1e-6,
) -> Dict[str, pd.DataFrame]:
    """
    For a multi-class target, compute term lift for each class vs rest.
    Returns dict of class_name -> DataFrame.
    """
    results = {}
    for cls in classes:
        pos_mask = labels == cls
        neg_mask = labels != cls

        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            continue

        pos_mean = tfidf_matrix[pos_mask].mean(axis=0)
        neg_mean = tfidf_matrix[neg_mask].mean(axis=0)

        valid = (pos_mean > min_mean) | (neg_mean > min_mean)
        lift = np.where(
            (neg_mean > min_mean) & valid,
            pos_mean / np.maximum(neg_mean, min_mean),
            0.0,
        )

        df = pd.DataFrame({
            "term": vocab,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "lift": lift,
            "log_lift": np.log2(np.maximum(lift, 1e-10)),
        })
        df = df[df["lift"] > 1.0].sort_values("lift", ascending=False).head(top_n)
        results[cls] = df.reset_index(drop=True)

    return results


# ============================================================
# Profanity analysis
# ============================================================

def count_profanity(text: str) -> int:
    """Count profanity instances in a text."""
    if not isinstance(text, str):
        return 0
    total = 0
    for pat in PROFANITY_PATTERNS:
        total += len(pat.findall(text))
    return total


def profanity_analysis(df: pd.DataFrame) -> Dict[str, any]:
    """Compute profanity rates overall and by each target."""
    print("  Scanning for profanity patterns...")

    df = df.copy()
    df["profanity_count"] = df["transcript_text"].fillna("").apply(count_profanity)
    df["has_profanity"] = (df["profanity_count"] > 0).astype(int)

    overall_rate = df["has_profanity"].mean()
    overall_avg_count = df["profanity_count"].mean()

    results = {
        "overall": {
            "total_docs": int(len(df)),
            "docs_with_profanity": int(df["has_profanity"].sum()),
            "profanity_rate": float(overall_rate),
            "avg_profanity_count": float(overall_avg_count),
        },
        "by_target": {},
    }

    # By each binary target
    for target_col, label in [
        ("target_proxy_gap", "proxy_vs_true_gap"),
        ("target_bandaid", "bandaid_credit"),
        ("target_repeat_30d", "repeat_contact_30d"),
    ]:
        if target_col in df.columns:
            pos = df[df[target_col] == 1]
            neg = df[df[target_col] == 0]
            results["by_target"][label] = {
                "positive_rate": float(pos["has_profanity"].mean()) if len(pos) > 0 else 0.0,
                "negative_rate": float(neg["has_profanity"].mean()) if len(neg) > 0 else 0.0,
                "positive_avg_count": float(pos["profanity_count"].mean()) if len(pos) > 0 else 0.0,
                "negative_avg_count": float(neg["profanity_count"].mean()) if len(neg) > 0 else 0.0,
                "positive_n": int(len(pos)),
                "negative_n": int(len(neg)),
            }

    # By scenario category
    for cat in ["clean", "gaming", "fraud", "ambiguous"]:
        subset = df[df["target_scenario_cat"] == cat]
        if len(subset) > 0:
            results["by_target"][f"scenario_{cat}"] = {
                "rate": float(subset["has_profanity"].mean()),
                "avg_count": float(subset["profanity_count"].mean()),
                "n": int(len(subset)),
            }

    return results


# ============================================================
# Charts
# ============================================================

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": CHART_BG,
        "axes.facecolor": "#16213e",
        "axes.edgecolor": CHART_FG,
        "axes.labelcolor": CHART_FG,
        "text.color": CHART_FG,
        "xtick.color": CHART_FG,
        "ytick.color": CHART_FG,
        "grid.color": "#2a2a4a",
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def chart_term_lift(
    df: pd.DataFrame, title: str, outpath: Path,
    top_n: int = 25, color: str = CHART_ACCENT,
) -> None:
    """Horizontal bar chart of top terms by lift."""
    if df.empty:
        return
    _apply_dark_style()

    d = df.head(top_n).copy().sort_values("log_lift", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(d) + 1)))
    ax.barh(d["term"], d["log_lift"], color=color, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Log\u2082 Lift")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=CHART_DPI)
    plt.close()


def chart_scenario_terms_panel(
    scenario_results: Dict[str, pd.DataFrame], out_dir: Path, top_n: int = 15,
) -> None:
    """Panel of term lift charts, one per scenario category."""
    cats = [c for c in ["clean", "gaming", "fraud"] if c in scenario_results]
    if not cats:
        return
    _apply_dark_style()

    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), max(5, 0.4 * top_n + 1)))
    if len(cats) == 1:
        axes = [axes]

    colors = {"clean": CHART_ACCENT, "gaming": CHART_WARN, "fraud": "#ffd93d"}

    for ax, cat in zip(axes, cats):
        df = scenario_results[cat].head(top_n).sort_values("log_lift", ascending=True)
        if df.empty:
            ax.set_title(f"{cat} (no data)")
            continue
        ax.barh(df["term"], df["log_lift"], color=colors.get(cat, CHART_ACCENT),
                alpha=0.85, edgecolor="none")
        ax.set_xlabel("Log\u2082 Lift")
        ax.set_title(f"Distinctive terms: {cat}")
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "scenario_terms_panel.png", dpi=CHART_DPI)
    plt.close()


def chart_profanity_rates(prof_results: Dict, out_dir: Path) -> None:
    """Bar chart of profanity rates by target."""
    _apply_dark_style()

    labels = []
    pos_rates = []
    neg_rates = []

    for target, stats in prof_results.get("by_target", {}).items():
        if "positive_rate" in stats:
            labels.append(target.replace("_", "\n"))
            pos_rates.append(stats["positive_rate"] * 100)
            neg_rates.append(stats["negative_rate"] * 100)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w / 2, pos_rates, w, label="Positive (target=1)", color=CHART_WARN, alpha=0.85)
    ax.bar(x + w / 2, neg_rates, w, label="Negative (target=0)", color=CHART_ACCENT, alpha=0.85)

    ax.set_xlabel("Target")
    ax.set_ylabel("% with profanity")
    ax.set_title("Profanity Rate by Outcome Target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(facecolor="#16213e", edgecolor=CHART_FG)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "profanity_rates_by_target.png", dpi=CHART_DPI)
    plt.close()


# ============================================================
# Reports
# ============================================================

def write_term_report(
    target_name: str,
    df: pd.DataFrame,
    outpath: Path,
    pos_label: str = "positive",
    neg_label: str = "negative",
) -> None:
    """Write a markdown report of term lift results."""
    lines = [
        f"# Term Lift Analysis: {target_name}",
        f"",
        f"**Positive class ({pos_label}):** terms overrepresented in this group vs rest",
        f"",
        f"| Rank | Term | Lift | Log\u2082 Lift | Pos Mean | Neg Mean |",
        f"|------|------|------|-----------|----------|----------|",
    ]
    for i, row in df.iterrows():
        lines.append(
            f"| {i+1} | {row['term']} | {row['lift']:.2f} | "
            f"{row['log_lift']:.2f} | {row['pos_mean']:.4f} | {row['neg_mean']:.4f} |"
        )
    lines.append("")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(
    corpus_size: int,
    vocab_size: int,
    lift_results: Dict[str, pd.DataFrame],
    scenario_results: Dict[str, pd.DataFrame],
    prof_results: Dict,
    outpath: Path,
) -> None:
    """Write the master summary report."""

    def pct(x):
        return f"{100 * x:.1f}%"

    lines = [
        "NLP TRUST SIGNALS PIPELINE — SUMMARY REPORT",
        "=" * 55,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Documents analyzed: {corpus_size:,}",
        f"Vocabulary size: {vocab_size:,} terms",
        "",
    ]

    # Per-target summaries
    for target_name, df in lift_results.items():
        if df.empty:
            continue
        lines.append(f"TOP 10 DISTINCTIVE TERMS: {target_name}")
        for i, row in df.head(10).iterrows():
            lines.append(f"  {i+1:>2}. {row['term']:<30} lift={row['lift']:.2f}")
        lines.append("")

    # Scenario terms
    if scenario_results:
        for cat, df in scenario_results.items():
            if df.empty:
                continue
            lines.append(f"TOP 10 DISTINCTIVE TERMS: scenario={cat}")
            for i, row in df.head(10).iterrows():
                lines.append(f"  {i+1:>2}. {row['term']:<30} lift={row['lift']:.2f}")
            lines.append("")

    # Profanity
    prof_overall = prof_results.get("overall", {})
    lines.append("PROFANITY ANALYSIS")
    lines.append(f"  Overall rate: {pct(prof_overall.get('profanity_rate', 0))}")
    lines.append(f"  Avg count per doc: {prof_overall.get('avg_profanity_count', 0):.2f}")
    lines.append("")

    for target, stats in prof_results.get("by_target", {}).items():
        if "positive_rate" in stats:
            lines.append(
                f"  {target}: positive={pct(stats['positive_rate'])} "
                f"(n={stats['positive_n']:,})  "
                f"negative={pct(stats['negative_rate'])} "
                f"(n={stats['negative_n']:,})"
            )
        elif "rate" in stats:
            lines.append(
                f"  {target}: rate={pct(stats['rate'])} (n={stats['n']:,})"
            )
    lines.append("")

    # Interpretation
    lines.append("INTERPRETATION GUIDE")
    lines.append("  - Lift > 1.0 means the term appears more often in the positive class.")
    lines.append("  - Log\u2082 lift of 1.0 means the term is 2x more frequent; 2.0 means 4x.")
    lines.append("  - proxy_vs_true_gap terms reveal the linguistic signature of calls where")
    lines.append("    the system claims resolution but the customer's problem persists.")
    lines.append("  - bandaid_credit terms reveal the language reps use when issuing")
    lines.append("    unauthorized credits to suppress repeat contacts.")
    lines.append("  - scenario category terms show what distinguishes gaming from fraud")
    lines.append("    from clean calls at the transcript level.")
    lines.append("  - Profanity rates by target show whether customer frustration (expressed")
    lines.append("    through language) correlates with system failure modes.")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Pipeline
# ============================================================

def cmd_run(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(Path.cwd())
    data_dir = repo_root / "data"
    out_dir = repo_root / "output"
    reports_dir = out_dir / "reports"
    figures_dir = out_dir / "figures"

    for d in [out_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("NLP TRUST SIGNALS PIPELINE")
    print("=" * 55)
    print(f"Repo root: {repo_root}")
    print("")

    # --- Load ---
    print("Loading monthly files...")
    df_raw = load_monthly_files(data_dir)

    # --- Prepare corpus ---
    print("Preparing corpus...")
    df = prepare_corpus(df_raw)

    # --- TF-IDF ---
    print("\nBuilding TF-IDF matrix...")
    vectorizer, tfidf_matrix = build_tfidf(
        df["clean_text"],
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 2),
    )
    vocab = vectorizer.get_feature_names_out()

    # --- Term lift: binary targets ---
    print("\nComputing term lift analysis...")

    binary_targets = {
        "proxy_vs_true_gap": ("target_proxy_gap", "proxy inflated", "genuinely resolved"),
        "bandaid_credit": ("target_bandaid", "bandaid issued", "no bandaid"),
        "repeat_contact_30d": ("target_repeat_30d", "repeat within 30d", "no repeat"),
    }

    lift_results = {}
    for target_name, (col, pos_label, neg_label) in binary_targets.items():
        if col not in df.columns:
            continue
        labels = df[col].values.astype(int)
        result = compute_term_lift_binary(
            tfidf_matrix, vocab, labels, top_n=args.top_n
        )
        lift_results[target_name] = result

        if not result.empty:
            # Save report
            write_term_report(
                target_name, result,
                reports_dir / f"terms_lift_{target_name}.md",
                pos_label=pos_label, neg_label=neg_label,
            )
            # Save chart
            color = CHART_WARN if "gap" in target_name or "bandaid" in target_name else CHART_ACCENT
            chart_term_lift(
                result,
                f"Distinctive terms: {target_name}",
                figures_dir / f"terms_lift_{target_name}.png",
                top_n=args.top_n,
                color=color,
            )
            print(f"  {target_name}: {len(result)} terms found (top lift={result['lift'].iloc[0]:.2f})")
        else:
            print(f"  {target_name}: insufficient data for analysis")

    # --- Term lift: scenario categories ---
    print("\nComputing scenario category term lift...")
    scenario_labels = df["target_scenario_cat"].values
    scenario_results = compute_term_lift_multiclass(
        tfidf_matrix, vocab, scenario_labels,
        classes=["clean", "gaming", "fraud"],
        top_n=args.top_n,
    )

    for cat, result in scenario_results.items():
        if not result.empty:
            write_term_report(
                f"scenario={cat}", result,
                reports_dir / f"terms_lift_scenario_{cat}.md",
                pos_label=cat, neg_label=f"not {cat}",
            )
            print(f"  scenario={cat}: {len(result)} terms (top lift={result['lift'].iloc[0]:.2f})")

    if scenario_results:
        chart_scenario_terms_panel(scenario_results, figures_dir, top_n=20)

    # --- Profanity analysis ---
    print("\nRunning profanity analysis...")
    prof_results = profanity_analysis(df)
    save_json(reports_dir / "profanity_analysis.json", prof_results)
    chart_profanity_rates(prof_results, figures_dir)

    overall_prof = prof_results.get("overall", {})
    print(f"  Overall profanity rate: {overall_prof.get('profanity_rate', 0):.1%}")
    print(f"  Avg count per doc: {overall_prof.get('avg_profanity_count', 0):.2f}")

    # --- Global top terms ---
    print("\nComputing global top terms...")
    total_tfidf = tfidf_matrix.sum(axis=0)
    top_idx = np.argsort(total_tfidf)[::-1][:args.top_n]
    global_terms = pd.DataFrame({
        "term": vocab[top_idx],
        "total_tfidf": total_tfidf[top_idx],
    })
    global_terms.to_csv(reports_dir / "top_terms_global.csv", index=False)

    # --- Summary report ---
    print("\nWriting summary report...")
    report_path = out_dir / "summary_report.txt"
    write_summary_report(
        corpus_size=len(df),
        vocab_size=len(vocab),
        lift_results=lift_results,
        scenario_results=scenario_results,
        prof_results=prof_results,
        outpath=report_path,
    )

    # --- Console summary ---
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print(f"Documents: {len(df):,}")
    print(f"Vocabulary: {len(vocab):,} terms")
    print("")
    print("Artifacts in output/:")
    print("  summary_report.txt")
    print("  reports/")
    for target_name in lift_results:
        print(f"    terms_lift_{target_name}.md")
    for cat in scenario_results:
        print(f"    terms_lift_scenario_{cat}.md")
    print("    profanity_analysis.json")
    print("    top_terms_global.csv")
    print("  figures/")
    for target_name in lift_results:
        print(f"    terms_lift_{target_name}.png")
    if scenario_results:
        print("    scenario_terms_panel.png")
    print("    profanity_rates_by_target.png")
    print("Done.")
    return 0


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NovaWireless NLP Trust Signals Pipeline (Lab-Aligned)"
    )
    p.add_argument("--max_features", type=int, default=25000,
                   help="Maximum TF-IDF vocabulary size")
    p.add_argument("--min_df", type=int, default=5,
                   help="Minimum document frequency for TF-IDF terms")
    p.add_argument("--max_df", type=float, default=0.85,
                   help="Maximum document frequency for TF-IDF terms")
    p.add_argument("--top_n", type=int, default=30,
                   help="Number of top terms to report per target")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
