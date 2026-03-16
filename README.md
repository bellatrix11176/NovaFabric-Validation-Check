# NovaFabric Validation Checklist

### Governance-grade proof that the friction signal is real — not a modeling artifact.

---

⚠️ Resource Notice
This pipeline is designed to be run iteratively on small datasets. Do not run against large datasets in a single pass. Process one month at a time to avoid memory and CPU overload. Adjust chunk sizes based on your available system resources.

---

Dashboards tell you what happened. This pipeline tells you whether you should believe it.

The NovaFabric Validation Checklist is a three-stage evidence pipeline that takes a synthetic call center dataset and produces audit-ready proof about the relationship between customer friction and operational outcomes. Every output is SHA-256 locked, run-stamped, and isolated — so when a reviewer asks "what data did you validate, what transforms did you apply, and could these outputs be stale," the answer is in the receipt chain, not in someone's memory.

This isn't a dashboard. It's a paper trail.

---

## What It Proves

Higher friction drives worse outcomes. The pipeline quantifies exactly how much worse, controls for confounders, and runs a placebo test to confirm it isn't finding signal that isn't there.

| Finding | Result |
|---|---|
| **Ticket odds** | 6.1× per SD increase in friction (95% CI 5.76–6.44) |
| **Resolution odds** | 0.19× per SD increase in friction — an 81% reduction |
| **Repeat-7D rate** | Monotonically increasing across all friction deciles |
| **Monotonicity** | Perfect 1.0 across all three outcome metrics |
| **Negative control** | OR ≈ 1.00 for both targets — no spurious signal detected |
| **Validation gate** | **PASS** |

The negative control is the line that separates this from a correlation exercise. A pseudo-random placebo variable, run through the same model specification, produces an odds ratio of 1.0 with confidence intervals that comfortably span 1.0. The friction signal survives. The placebo doesn't.

---

## Quick Start

```bash
pip install -r requirements.txt
python src/run_all.py
```

One command runs all three stages. Results land in `output/novafabric_validation/`.

```bash
python src/run_all.py --from 2     # skip instrumentation, resume from validation
python src/run_all.py --only 3     # run only the evidence pack
```

---

## The Three Stages

### Stage 1 — Instrumentation
**`src/00_instrument_novafabric.py`**

Takes raw call records and creates stable `*_filled` columns encoding a deterministic causal structure. Friction drives ticketing (positive), suppresses resolution (negative), and compounds through repeat contact and trust decay. All randomness is seeded. Every parameter is logged in `instrumentation_receipt.json`.

### Stage 2 — Validation
**`src/01_validate_novafabric.py`**

Coverage gates, range checks, missingness audit, correlation directionality verification, and time-series stability scans. Produces diagnostic plots and a human-readable markdown report. Does *not* fit models — that's Stage 3's job.

### Stage 3 — Evidence Pack
**`src/02_evidence_pack.py`**

The formal proof artifacts:

- **Decile lift table** — ticket, resolved, and repeat-7D rates across 10 friction bins with monotonicity scores
- **4 logistic regressions** — uncontrolled and controlled (subreason + rep_id) for ticket and resolved outcomes, with bootstrap confidence intervals (300 resamples)
- **Negative control test** — a pseudo-random placebo variable regressed against both outcomes using the same controlled specification

Every evidence run gets its own folder under `evidence/run_<runid>/` with a frozen copy of the instrumentation receipt and an independent SHA-256 hash of the input file.

---

## Integrity Chain

| Question a Reviewer Will Ask | Where the Answer Lives |
|---|---|
| What data did you validate? | `input_sha256.txt` |
| What transformations did you apply? | `instrumentation_receipt.json` |
| What did the run produce? | `evidence_summary.json` |
| Could these outputs be stale? | `RUNSTAMP_<run_id>.txt` in run-isolated directories |
| What was the reference category? | `evidence_summary.json` → `reference_categories` |

---

## Output Structure

```
output/novafabric_validation/
├── NovaFabric_instrumented.csv                 Instrumented dataset
├── instrumentation_receipt.json                 Stage 1 parameter receipt
├── validation_summary.json                      Stage 2 machine-readable summary
├── validation_report.md                         Stage 2 human-readable report
├── validation_metrics_by_bucket.csv             Rates by subreason
├── validation_metrics_by_rep.csv                Rates by representative
├── input_sha256.txt                             Tamper-evidence hash
├── plots/run_<runid>/*.png                      Diagnostic plots
│
└── evidence/run_<runid>/                        Stage 3 evidence pack
    ├── evidence_summary.json                    Run metadata + monotonicity scores
    ├── copied_instrumentation_receipt.json       Frozen Stage 1 receipt
    ├── input_sha256.txt                         Independent hash verification
    ├── decile_lift_table.csv                    Primary lift results
    ├── lift_ticket_rate.png                     Lift plot: ticket
    ├── lift_resolved_rate.png                   Lift plot: resolved
    ├── lift_repeat7_rate.png                    Lift plot: repeat-7D
    ├── logit_ticket_uncontrolled_oddsratios.csv
    ├── logit_ticket_controls_oddsratios.csv
    ├── logit_resolved_uncontrolled_oddsratios.csv
    ├── logit_resolved_controls_oddsratios.csv
    └── negative_control_results.csv             Placebo test
```

---

## Companion Paper

> Aulabaugh, G. (2026). *Governance-Grade Evidence for KPI Risk Under AI-Optimized Call Center Dynamics: NovaFabric Validation Checklist (Synthetic Case Study).*

---

## Requirements

Python 3.9+ with `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.

```bash
pip install -r requirements.txt
```

---

<p align="center">
  <b>Gina Aulabaugh</b><br>
  <a href="https://www.pixelkraze.com">www.pixelkraze.com</a>
</p>
