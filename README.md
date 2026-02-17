# NovaFabric Validation Check

Governance-grade validation pipeline for a synthetic call-center dataset (**NovaFabric**).  
This repo produces “paper-proof” artifacts: instrumentation receipts, input hashing, run-isolated plots, decile lift tables, and logistic regression odds ratios—so results are traceable, repeatable, and reviewer-friendly.

## What this project does

This pipeline turns a raw synthetic dataset into an **audit-style evidence chain**:

1. **Instrument** the dataset into stable, explicit “filled” columns (so missingness and naming drift don’t silently change results).
2. **Validate** core integrity gates (coverage/missingness, ranges/sanity checks, time-series stability).
3. **Prove relationships** with heavyweight artifacts:
   - **Friction decile lift** (ticket / resolved / repeat-7D rates by friction decile)
   - **Logistic regression odds ratios** (uncontrolled + controlled)

The goal is not “fancy ML.” The goal is **defensible evidence**.

---

## Repository Structure

```
NovaFabric-Validation-Check/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ citations.txt
├
|─ assets/
│  ├─ NovaFabric.png
│  ├─ NovaWirelessAIOperations.png
|  ├─ NovaWirelessLogo.png
│  └─ NovaWirelessPromoDesign.png
|
├─ data/
│  ├─ 00_instrument_novafabric.py
│  ├─ 01_validate_novafabric.py
│  └─ 02_friction_lift_and_models.pyv
│
├─ src/
│  ├─ 00_instrument_novafabric.py
│  ├─ 01_validate_novafabric.py
│  └─ 02_friction_lift_and_models.py
│
├─ Papers/
│  ├─ Governance_Grade_Evidence_for_KPI_Risk_Under_AI_Optimized_Call_Center_Dynamics.pdf
│  └─ When_KPIs_Lie__Governance_Signals_for_AI_Optimized_Call_Centers.pdf
│
└─ output/
   └─ novafabric_validation/
      ├─ NovaFabric_instrumented.csv
      ├─ instrumentation_receipt.json
      ├─ input_sha256.txt
      ├─ model_summary.json
      ├─ decile_lift_table.csv
      ├─ logit_ticket_uncontrolled_oddsratios.csv
      ├─ logit_ticket_controls_oddsratios.csv
      ├─ logit_resolved_uncontrolled_oddsratios.csv
      ├─ logit_resolved_controls_oddsratios.csv
      ├─ validation_metrics_by_bucket.csv
      ├─ validation_metrics_by_rep.csv
      ├─ validation_report.md
      ├─ validation_summary.json
      │
      ├─ plots/
      │  ├─ run_acdbbb4123/
      │  │  ├─ hist_friction.png
      │  │  ├─ hist_trust.png
      │  │  ├─ ts_calls.png
      │  │  ├─ ts_friction_mean.png
      │  │  ├─ ts_resolved_rate.png
      │  │  └─ ts_ticket_rate.png
      │  └─ run_e7d11eb4f5/
      │     ├─ hist_friction.png
      │     ├─ hist_trust.png
      │     ├─ lift_repeat7_rate.png
      │     ├─ lift_resolved_rate.png
      │     ├─ lift_ticket_rate.png
      │     ├─ ts_calls.png
      │     ├─ ts_friction_mean.png
      │     ├─ ts_resolved_rate.png
      │     └─ ts_ticket_rate.png
      │
      ├─ paper_proof/
      │  ├─ friction_decile_lift.csv
      │  ├─ plots/
      │  │  ├─ lift_repeat7_rate.png
      │  │  ├─ lift_resolved_rate.png
      │  │  └─ lift_ticket_rate.png
      │  └─ models/
      │     ├─ logit_ticket_oddsratios.csv
      │     ├─ logit_resolved_oddsratios.csv
      │     └─ model_summary.json
      │
      └─ evidence/
         └─ run_4a78f23bdb/
            ├─ RUNSTAMP_4a78f23bdb.txt
            ├─ input_sha256.txt
            ├─ copied_instrumentation_receipt.json
            ├─ evidence_summary.json
            ├─ decile_lift_table.csv
            ├─ negative_control_results.csv
            ├─ lift_repeat7_rate.png
            ├─ lift_resolved_rate.png
            ├─ lift_ticket_rate.png
            ├─ logit_ticket_uncontrolled_oddsratios.csv
            ├─ logit_ticket_controls_oddsratios.csv
            ├─ logit_resolved_uncontrolled_oddsratios.csv
            └─ logit_resolved_controls_oddsratios.csv
```
