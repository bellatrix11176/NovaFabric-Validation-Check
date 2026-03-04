# NovaFabric Validation Checklist

- Status: **PASS**
- Run ID: `e7d11eb4f5`
- Generated: 2026-02-16T20:01:25
- Script: `C:\Users\gigih\OneDrive\NovaWireless\NovaFabric Validation Checklist\src\01_validate_novafabric.py`
- Input: `output/novafabric_validation/NovaFabric_instrumented.csv`
- Input SHA256: `bfd02a87d55f8e79610ff8748c1ddbd9a45a040851385d58bf6eef081e06a91b`
- Rows: **20,000** | Columns: **32**

## Key Rates
- Resolved rate: **39.0%**
- Ticket rate: **35.9%**
- Repeat-7D rate: **32.7%**

## Directionality Checks (Correlations)
- friction vs trust: **-0.663 (n=20000)** (expect negative)
- friction vs resolved: **-0.461 (n=20000)** (often negative)
- friction vs ticket: **0.495 (n=20000)** (expect positive)
- friction vs repeat_7d: **0.430 (n=20000)** (often positive)

## Evidence Pack A — Friction Decile Lift
Saved: `decile_lift_table.csv` and lift plots in the run plots folder.
|   friction_decile |   calls |   friction_mean |   ticket_rate |   resolved_rate |   repeat7_rate |
|------------------:|--------:|----------------:|--------------:|----------------:|---------------:|
|                 1 |    2000 |        0.235026 |        0.026  |          0.8605 |         0.0455 |
|                 2 |    2000 |        0.29883  |        0.095  |          0.6605 |         0.1145 |
|                 3 |    2000 |        0.318781 |        0.1575 |          0.541  |         0.182  |
|                 4 |    2000 |        0.333038 |        0.2205 |          0.4745 |         0.2045 |
|                 5 |    2000 |        0.345373 |        0.2615 |          0.413  |         0.246  |
|                 6 |    2000 |        0.357365 |        0.334  |          0.332  |         0.318  |
|                 7 |    2000 |        0.370183 |        0.4435 |          0.265  |         0.3825 |
|                 8 |    2000 |        0.385515 |        0.519  |          0.2    |         0.4545 |
|                 9 |    2000 |        0.407877 |        0.657  |          0.1255 |         0.544  |
|                10 |    2000 |        0.484723 |        0.881  |          0.0315 |         0.779  |

## Evidence Pack B — Logistic Regression (4 models)
Odds ratios are reported for **friction_z** (1 SD increase).
- Ticket (uncontrolled): OR=6.070 (95% CI 5.716–6.446) per +1 SD friction
- Ticket (+ subreason, rep controls): OR=6.111 (95% CI 5.753–6.491) per +1 SD friction
- Resolved (uncontrolled): OR=0.194 (95% CI 0.184–0.206) per +1 SD friction
- Resolved (+ subreason, rep controls): OR=0.194 (95% CI 0.183–0.205) per +1 SD friction

Full metadata saved to `model_summary.json`.

## Coverage Gate
- Fail threshold: missing fraction > 5.0%
- friction_level_filled: missing 0.0%
- trust_score_filled: missing 0.0%
- ticket_flag_filled: missing 0.0%
- resolved_flag_filled: missing 0.0%
- ✅ PASS

## Warnings
- ⚠️ High-missingness columns (>= 20.0%): ['outcome', 'friction_level']

## Plots
- Run plots folder: `output\novafabric_validation\plots\run_e7d11eb4f5`
- hist_friction.png
- hist_trust.png
- ts_calls.png / ts_friction_mean.png / ts_ticket_rate.png / ts_resolved_rate.png
- lift_ticket_rate.png / lift_resolved_rate.png
- lift_repeat7_rate.png
