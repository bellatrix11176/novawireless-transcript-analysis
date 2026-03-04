# novawireless-transcript-analysis

### The transcripts already contain the evidence. This pipeline extracts it.

---

Your agents are telling you what's wrong — in the language they use to avoid fixing it.

When a rep says "seven business days" instead of resolving the issue, that's a deflection marker. When "courtesy credit" appears in a call labeled "resolved" but the customer calls back 45 days later, that's a gaming signature. When profanity spikes in fraud scenarios but not clean ones, that's frustration correlated with institutional failure, not random noise.

This pipeline reads 82,000+ call transcripts and identifies the exact terms that distinguish gamed calls from genuine ones, fraud from clean resolution, and bandaid credits from authorized fixes — using TF-IDF term lift analysis. No black-box models. No embeddings you can't explain to a regulator. Just term frequencies, lift ratios, and a clear answer: **which words predict that the KPI is lying?**

---

## What It Finds

| Target | Top Term | Lift | What It Means |
|---|---|---|---|
| Gaming scenario | "seven business" | 50.0× | Deflection language — stalling instead of fixing |
| Bandaid credit | "works" | 47.3× | Call-closing language after an unauthorized credit |
| Fraud scenario | "store" | 35.1× | Store interaction disputes — the customer didn't authorize it |
| Clean scenario | "IMEI" | 22.9× | Technical resolution language — the rep actually fixed it |
| Repeat contact | "team typically" | 3.9× | Escalation language — system failure, not agent gaming |

A 50× lift means that term appears 50 times more frequently in gaming calls than in the general population. That's not a correlation. That's a linguistic fingerprint.

---

## Five Analyses, One Pipeline

### 1. Scenario Category Signatures
One-versus-rest term lift across clean, gaming, and fraud call categories. Gaming terms cluster around deflection and delay language. Fraud terms cluster around dispute and unauthorized access language. Clean terms cluster around technical resolution language. The three vocabularies are almost entirely non-overlapping.

### 2. Proxy-Versus-True Gap
Terms that appear when the proxy says "resolved" but the customer's problem persists. Top terms include NRF-related fraud language and empathetic closure phrases — the linguistic footprint of a call that *sounds* resolved but isn't.

### 3. Bandaid Credit Signatures
Terms associated with unauthorized credit issuance. Produces the highest single-term lifts in the entire analysis (47×). 28 of the top 30 terms share identical lift values — reflecting concentration in a single scenario template. That homogeneity is itself a detection signal.

### 4. Repeat Contact Predictors
Terms that predict 30-day callbacks. These cluster around failed activation and escalation language, indicating that repeat contacts are driven by system failures rather than agent gaming — a finding that changes where you point your intervention.

### 5. Profanity Analysis
Profanity rates by outcome target. Customer frustration correlates with proxy-true divergence and gaming scenarios. Customers whose problems are deflected swear more than customers whose problems are solved. The frustration signal is real, and it's in the transcript.

---

## Quick Start

```bash
pip install -r requirements.txt
python novawireless_nlp_trust_signals.py
```

Copy the 12 monthly `calls_sanitized_2025-*.csv` files from the NovaWireless Call Center Lab into `data/`. Runtime is 2–4 minutes on 82,000+ transcripts.

The pipeline handles everything: PII redaction, whitespace normalization, dollar amount tokenization, TF-IDF vectorization (sublinear TF, unigrams + bigrams, max 25,000 terms), term lift computation, figure generation, and JSON report output.

---

## What It Produces

```
output/
├── scenario_term_lift.png              Clean / gaming / fraud term lift panel
├── proxy_gap_term_lift.png             Proxy-vs-true gap terms
├── bandaid_term_lift.png               Bandaid credit terms
├── repeat_contact_term_lift.png        Repeat contact terms
├── profanity_rates_by_target.png       Profanity rates by outcome target
├── profanity_analysis.json             Profanity scan results
└── nlp_analysis_report.json            Full analysis summary
```

---

## Why This Complements the Governance Pipeline

The governance pipeline detects proxy-outcome divergence through **structured metadata** — credit types, resolution flags, detection signals. This pipeline detects the same divergence through **transcript language**.

Two completely independent signal families arriving at the same conclusion: the proxy is broken. That convergence is what makes the evidence governance-grade. A single detection method can be argued away. Two independent methods producing the same result cannot.

| Repository | Signal Family | Detection Method |
|---|---|---|
| novawireless-governance-pipeline | Structured metadata | Trust scoring, threshold alerts |
| **novawireless-transcript-analysis** | **Transcript language** | **TF-IDF term lift, profanity analysis** |
| NovaWireless_KPI_Drift_Observatory | Composite integrity | SII framework |
| NovaFabric Validation Checklist | Causal validation | Friction lift, logistic models |

---

## Repository Structure

```
novawireless-transcript-analysis/
├── novawireless_nlp_trust_signals.py   Main pipeline (~837 lines)
├── transcript_analysis_paper.pdf       Companion paper with 5 embedded figures
├── data/                               Input CSVs (not committed)
├── output/                             Pipeline outputs (gitignored)
└── README.md
```

---

## Companion Paper

> Aulabaugh, G. (2026). *Linguistic Signatures of Proxy-Outcome Divergence in Synthetic Call Center Transcripts: A TF-IDF Term Lift Analysis of the NovaWireless Call Center Lab.* PixelKraze LLC.

---

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`.

---

<p align="center">
  <b>Gina Aulabaugh</b><br>
  <a href="https://www.pixelkraze.com">www.pixelkraze.com</a>
</p>
