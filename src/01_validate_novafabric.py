#!/usr/bin/env python3
"""
01_validate_novafabric.py

NovaFabric Validation Checklist (Validator + Evidence Pack)

Repo-root compliant: run from anywhere.

Reads (default):
- output/novafabric_validation/NovaFabric_instrumented.csv

Writes:
- output/novafabric_validation/validation_summary.json
- output/novafabric_validation/validation_report.md
- output/novafabric_validation/validation_metrics_by_bucket.csv
- output/novafabric_validation/validation_metrics_by_rep.csv

Evidence artifacts (paper-proof):
- output/novafabric_validation/decile_lift_table.csv
- output/novafabric_validation/logit_ticket_uncontrolled_oddsratios.csv
- output/novafabric_validation/logit_ticket_controls_oddsratios.csv
- output/novafabric_validation/logit_resolved_uncontrolled_oddsratios.csv
- output/novafabric_validation/logit_resolved_controls_oddsratios.csv
- output/novafabric_validation/model_summary.json

Plots (per run to prevent stale overwrite / OneDrive caching):
- output/novafabric_validation/plots/run_<run_id>/*.png
- output/novafabric_validation/plots/run_<run_id>/RUNSTAMP_<run_id>.txt
- output/novafabric_validation/RUNSTAMP_<run_id>.txt
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import hashlib

import pandas as pd

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")  # must be set before pyplot
import matplotlib.pyplot as plt


# -----------------------------
# Repo-root helpers
# -----------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    return Path(__file__).resolve().parent.parent


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    input_csv: str = "output/novafabric_validation/NovaFabric_instrumented.csv"
    out_dir: str = "output/novafabric_validation"

    # warnings / sanity checks
    max_missing_frac_warn: float = 0.20
    friction_range: Tuple[float, float] = (0.0, 1.0)
    trust_range: Tuple[float, float] = (0.0, 100.0)

    warn_if_resolved_rate_lt: float = 0.25
    warn_if_resolved_rate_gt: float = 0.85
    warn_if_ticket_rate_lt: float = 0.01
    warn_if_ticket_rate_gt: float = 0.60

    rep_min_calls_for_outlier: int = 30
    outlier_z: float = 2.5

    # coverage gate (optional hard fail)
    fail_if_missing_frac_gt: float = 0.05
    critical_cols: Tuple[str, ...] = (
        "friction_level_filled",
        "trust_score_filled",
        "ticket_flag_filled",
        "resolved_flag_filled",
    )
    hard_fail: bool = False

    # evidence pack
    friction_deciles: int = 10
    max_levels_controls: int = 50   # cap high-cardinality controls to avoid model blow-ups
    include_rep_controls: bool = True


# -----------------------------
# Utility
# -----------------------------
def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100.0 * float(x):.1f}%"


def is_nan(x: Any) -> bool:
    try:
        return x != x
    except Exception:
        return False


def to_int_flag(series: pd.Series) -> pd.Series:
    def conv(v: Any) -> int:
        if v is None or is_nan(v):
            return 0
        try:
            return int(float(v))
        except Exception:
            s = str(v).strip().lower()
            return 1 if s in ("true", "yes", "y", "1") else 0
    return series.apply(conv).astype(int)


def ensure_col(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


def corr_safe_with_n(df: pd.DataFrame, a: str, b: str) -> Tuple[Optional[float], int]:
    if a not in df.columns or b not in df.columns:
        return None, 0
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return None, n
    return float(x[m].corr(y[m])), n


def z_scores(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def pick_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def cap_top_levels(s: pd.Series, max_levels: int, other_label: str = "__OTHER__") -> pd.Series:
    s = s.fillna(other_label).astype(str)
    vc = s.value_counts(dropna=False)
    keep = set(vc.head(max_levels).index.tolist())
    return s.apply(lambda x: x if x in keep else other_label)


# -----------------------------
# Plotting
# -----------------------------
def plot_hist(series: pd.Series, title: str, out_path: Path, bins: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(series.name if series.name else "value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_time_series(df: pd.DataFrame, date_col: str, y_col: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df[[date_col, y_col]].dropna()
    if d.empty:
        return
    plt.figure()
    plt.plot(d[date_col], d[y_col])
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_lift_line(x: pd.Series, y: pd.Series, title: str, ylab: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("friction decile (1=lowest)")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Logistic regression helpers (statsmodels)
# -----------------------------
def fit_logit_odds_ratios(
    df: pd.DataFrame,
    y_col: str,
    friction_col: str,
    controls: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits: y ~ friction_z (+ controls as categorical via C()).

    Returns:
      - odds ratio table for friction_z with CI
      - metadata summary dict
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.copy()

    # outcome must be 0/1
    d[y_col] = to_int_flag(d[y_col])

    # friction z-score (1 SD step)
    friction_num = pd.to_numeric(d[friction_col], errors="coerce")
    friction_z = (friction_num - friction_num.mean()) / (friction_num.std(ddof=0) if friction_num.std(ddof=0) != 0 else 1.0)
    d["friction_z"] = friction_z

    base_cols = [y_col, "friction_z"]
    if controls:
        base_cols += controls

    d = d[base_cols].dropna().copy()
    n = int(len(d))
    if n < 50:
        # too small -> return empty, but don't crash your pipeline
        out = pd.DataFrame([{"feature": "friction_z", "odds_ratio": None, "ci_2p5": None, "ci_97p5": None}])
        meta = {"n_used": n, "status": "SKIP_TOO_SMALL"}
        return out, meta

    # formula
    formula = f"{y_col} ~ friction_z"
    if controls:
        # treat controls as categorical
        for c in controls:
            formula += f" + C({c})"

    model = smf.logit(formula=formula, data=d)
    res = model.fit(disp=False, maxiter=200)

    # CI for friction_z
    params = res.params
    conf = res.conf_int()
    if "friction_z" not in params.index:
        out = pd.DataFrame([{"feature": "friction_z", "odds_ratio": None, "ci_2p5": None, "ci_97p5": None}])
    else:
        beta = float(params["friction_z"])
        lo = float(conf.loc["friction_z", 0])
        hi = float(conf.loc["friction_z", 1])
        out = pd.DataFrame([{
            "feature": "friction_z",
            "odds_ratio": math.exp(beta),
            "ci_2p5": math.exp(lo),
            "ci_97p5": math.exp(hi),
        }])

    meta = {
        "n_used": n,
        "status": "OK",
        "pseudo_r2_mcfadden": float(getattr(res, "prsquared", float("nan"))),
        "aic": float(getattr(res, "aic", float("nan"))),
        "bic": float(getattr(res, "bic", float("nan"))),
        "llf": float(getattr(res, "llf", float("nan"))),
        "formula": formula,
    }
    return out, meta


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = Config()
    root = find_repo_root()

    in_path = root / cfg.input_csv
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input CSV:\n  {in_path}\n\n"
            "Fix: run src/00_instrument_novafabric.py OR update Config.input_csv."
        )

    out_dir = root / cfg.out_dir
    base_plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_plots_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex[:10]
    now_ts = datetime.now().isoformat(timespec="seconds")

    run_plots_dir = base_plots_dir / f"run_{run_id}"
    run_plots_dir.mkdir(parents=True, exist_ok=True)

    # hard proof (audit receipts)
    safe_write_text(out_dir / f"RUNSTAMP_{run_id}.txt", now_ts)
    safe_write_text(run_plots_dir / f"RUNSTAMP_{run_id}.txt", now_ts)
    safe_write_text(out_dir / "input_sha256.txt", sha256_file(in_path))

    print("\n=== NovaFabric Validator Run ===")
    print("RUN_ID     :", run_id)
    print("NOW        :", now_ts)
    print("SCRIPT     :", Path(__file__).resolve())
    print("REPO_ROOT  :", root)
    print("INPUT_CSV  :", in_path)
    print("OUT_DIR    :", out_dir)
    print("PLOTS_DIR  :", run_plots_dir)
    print("================================\n")

    df = pd.read_csv(in_path, low_memory=False)

    # baseline columns
    ensure_col(df, "interaction_id", "")
    ensure_col(df, "customer_id", "")
    ensure_col(df, "rep_id", "")
    ensure_col(df, "timestamp", "")
    ensure_col(df, "subreason_std", df["subreason"] if "subreason" in df.columns else "Other")

    # prefer *_filled
    friction_col = pick_col(df, "friction_level_filled", "friction_level")
    trust_col = pick_col(df, "trust_score_filled", "trust_score")
    ticket_col = pick_col(df, "ticket_flag_filled", "ticket_flag")
    resolved_col = pick_col(df, "resolved_flag_filled", "resolved_flag")
    repeat7_col = pick_col(df, "repeat_7d_flag_filled", "repeat_7d_flag")

    # ensure fallbacks
    ensure_col(df, "friction_level", 0.35)
    ensure_col(df, "resolved_flag", 0)
    ensure_col(df, "ticket_flag", 0)

    # coerce flags
    df[ticket_col] = to_int_flag(df[ticket_col]) if ticket_col in df.columns else 0
    df[resolved_col] = to_int_flag(df[resolved_col]) if resolved_col in df.columns else 0
    if repeat7_col in df.columns:
        df[repeat7_col] = to_int_flag(df[repeat7_col])

    # timestamps
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp_dt"].dt.date

    n = int(len(df))

    # presence flags
    has_trust = trust_col in df.columns and pd.to_numeric(df[trust_col], errors="coerce").notna().any()
    has_repeat7 = repeat7_col in df.columns

    # missingness + coverage gate
    missingness = (df.isna().mean()).sort_values(ascending=False)
    high_missing_cols = [c for c, frac in missingness.items() if frac >= cfg.max_missing_frac_warn]

    crit_missing: Dict[str, float] = {}
    for c in cfg.critical_cols:
        crit_missing[c] = float(df[c].isna().mean()) if c in df.columns else 1.0

    hard_fail_reasons = [f"{k} missing {pct(v)}" for k, v in crit_missing.items() if v > cfg.fail_if_missing_frac_gt]
    status = "PASS" if not hard_fail_reasons else "FAIL"

    # range checks
    friction_num = pd.to_numeric(df[friction_col], errors="coerce")
    friction_oob = int(((friction_num < cfg.friction_range[0]) | (friction_num > cfg.friction_range[1])).sum())

    trust_oob = None
    if has_trust:
        trust_num = pd.to_numeric(df[trust_col], errors="coerce")
        trust_oob = int(((trust_num < cfg.trust_range[0]) | (trust_num > cfg.trust_range[1])).sum())

    # rates
    resolved_rate = float(pd.to_numeric(df[resolved_col], errors="coerce").fillna(0).mean()) if n else 0.0
    ticket_rate = float(pd.to_numeric(df[ticket_col], errors="coerce").fillna(0).mean()) if n else 0.0
    repeat7_rate = float(pd.to_numeric(df[repeat7_col], errors="coerce").fillna(0).mean()) if (n and has_repeat7) else None

    # correlations
    corr_map: Dict[str, Tuple[Optional[float], int]] = {
        "corr_friction_trust": corr_safe_with_n(df, friction_col, trust_col) if has_trust else (None, 0),
        "corr_friction_resolved": corr_safe_with_n(df, friction_col, resolved_col),
        "corr_friction_ticket": corr_safe_with_n(df, friction_col, ticket_col),
        "corr_friction_repeat7": corr_safe_with_n(df, friction_col, repeat7_col) if has_repeat7 else (None, 0),
    }
    corrs = {k: v for k, (v, _) in corr_map.items()}
    corr_n = {k: n_ for k, (_, n_) in corr_map.items()}

    # time texture
    daily = pd.DataFrame()
    time_summary = None
    if df["timestamp_dt"].notna().any():
        daily = (
            df.groupby("date")
            .agg(
                calls=("interaction_id", "count"),
                friction_mean=(friction_col, "mean"),
                resolved_rate=(resolved_col, "mean"),
                ticket_rate=(ticket_col, "mean"),
            )
            .reset_index()
        )
        time_summary = {
            "days": int(len(daily)),
            "min_date": str(daily["date"].min()) if len(daily) else None,
            "max_date": str(daily["date"].max()) if len(daily) else None,
        }

    # -----------------------------
    # Plots: hist + time series
    # -----------------------------
    plot_hist(friction_num, f"Friction distribution ({friction_col})", run_plots_dir / "hist_friction.png", bins=40)
    if has_trust:
        plot_hist(
            pd.to_numeric(df[trust_col], errors="coerce"),
            f"Trust distribution ({trust_col})",
            run_plots_dir / "hist_trust.png",
            bins=40,
        )
    if not daily.empty:
        plot_time_series(daily, "date", "calls", "Daily call volume", run_plots_dir / "ts_calls.png")
        plot_time_series(daily, "date", "friction_mean", f"Daily mean friction ({friction_col})", run_plots_dir / "ts_friction_mean.png")
        plot_time_series(daily, "date", "ticket_rate", f"Daily ticket rate ({ticket_col})", run_plots_dir / "ts_ticket_rate.png")
        plot_time_series(daily, "date", "resolved_rate", f"Daily resolved rate ({resolved_col})", run_plots_dir / "ts_resolved_rate.png")

    # -----------------------------
    # Evidence 1: friction decile lift table + plots
    # -----------------------------
    d_ev = df.copy()
    d_ev = d_ev[pd.to_numeric(d_ev[friction_col], errors="coerce").notna()].copy()
    d_ev["friction_num"] = pd.to_numeric(d_ev[friction_col], errors="coerce")

    # deciles (1..10)
    try:
        d_ev["friction_decile"] = pd.qcut(d_ev["friction_num"], q=cfg.friction_deciles, labels=False, duplicates="drop") + 1
    except Exception:
        # fallback if qcut struggles (e.g., too many ties)
        d_ev["friction_decile"] = pd.cut(d_ev["friction_num"], bins=cfg.friction_deciles, labels=False, include_lowest=True) + 1

    lift_cols = {
        "calls": ("interaction_id", "count"),
        "friction_mean": ("friction_num", "mean"),
        "ticket_rate": (ticket_col, "mean"),
        "resolved_rate": (resolved_col, "mean"),
    }
    if has_repeat7:
        lift_cols["repeat7_rate"] = (repeat7_col, "mean")

    lift = (
        d_ev.groupby("friction_decile")
        .agg(**lift_cols)
        .reset_index()
        .sort_values("friction_decile")
    )

    lift.to_csv(out_dir / "decile_lift_table.csv", index=False)

    # lift plots
    plot_lift_line(lift["friction_decile"], lift["ticket_rate"], "Ticket rate by friction decile", "ticket_rate", run_plots_dir / "lift_ticket_rate.png")
    plot_lift_line(lift["friction_decile"], lift["resolved_rate"], "Resolved rate by friction decile", "resolved_rate", run_plots_dir / "lift_resolved_rate.png")
    if has_repeat7:
        plot_lift_line(lift["friction_decile"], lift["repeat7_rate"], "Repeat-7D rate by friction decile", "repeat7_rate", run_plots_dir / "lift_repeat7_rate.png")

    # -----------------------------
    # Evidence 2: 4 logistic models (uncontrolled + controlled)
    # -----------------------------
    # controls (cap top levels to keep model stable)
    controls: List[str] = []
    df["subreason_std"] = cap_top_levels(df["subreason_std"], cfg.max_levels_controls)
    controls.append("subreason_std")

    if cfg.include_rep_controls:
        df["rep_id"] = cap_top_levels(df["rep_id"], cfg.max_levels_controls)
        controls.append("rep_id")

    # Ticket models
    ticket_unctrl, meta_ticket_unctrl = fit_logit_odds_ratios(
        df=df,
        y_col=ticket_col,
        friction_col=friction_col,
        controls=None,
    )
    ticket_ctrl, meta_ticket_ctrl = fit_logit_odds_ratios(
        df=df,
        y_col=ticket_col,
        friction_col=friction_col,
        controls=controls,
    )

    # Resolved models
    resolved_unctrl, meta_res_unctrl = fit_logit_odds_ratios(
        df=df,
        y_col=resolved_col,
        friction_col=friction_col,
        controls=None,
    )
    resolved_ctrl, meta_res_ctrl = fit_logit_odds_ratios(
        df=df,
        y_col=resolved_col,
        friction_col=friction_col,
        controls=controls,
    )

    ticket_unctrl.to_csv(out_dir / "logit_ticket_uncontrolled_oddsratios.csv", index=False)
    ticket_ctrl.to_csv(out_dir / "logit_ticket_controls_oddsratios.csv", index=False)
    resolved_unctrl.to_csv(out_dir / "logit_resolved_uncontrolled_oddsratios.csv", index=False)
    resolved_ctrl.to_csv(out_dir / "logit_resolved_controls_oddsratios.csv", index=False)

    model_summary = {
        "generated_at": now_ts,
        "run_id": run_id,
        "friction_col": friction_col,
        "outcomes": {"ticket": ticket_col, "resolved": resolved_col},
        "controls_used": controls,
        "models": {
            "ticket_uncontrolled": meta_ticket_unctrl,
            "ticket_controls": meta_ticket_ctrl,
            "resolved_uncontrolled": meta_res_unctrl,
            "resolved_controls": meta_res_ctrl,
        },
    }
    safe_write_json(out_dir / "model_summary.json", model_summary)

    # -----------------------------
    # Existing tables
    # -----------------------------
    by_bucket = (
        df.groupby("subreason_std", dropna=False)
        .agg(
            calls=("interaction_id", "count"),
            friction_mean=(friction_col, "mean"),
            ticket_rate=(ticket_col, "mean"),
            resolved_rate=(resolved_col, "mean"),
        )
        .sort_values("calls", ascending=False)
        .reset_index()
    )

    by_rep = (
        df.groupby("rep_id", dropna=False)
        .agg(
            calls=("interaction_id", "count"),
            friction_mean=(friction_col, "mean"),
            ticket_rate=(ticket_col, "mean"),
            resolved_rate=(resolved_col, "mean"),
        )
        .sort_values("calls", ascending=False)
        .reset_index()
    )

    by_bucket.to_csv(out_dir / "validation_metrics_by_bucket.csv", index=False)
    by_rep.to_csv(out_dir / "validation_metrics_by_rep.csv", index=False)

    # warnings
    warnings: List[str] = []
    if hard_fail_reasons:
        warnings.append(f"HARD_FAIL: {hard_fail_reasons}")
    if friction_oob > 0:
        warnings.append(f"Friction out-of-bounds rows: {friction_oob}")
    if trust_oob is not None and trust_oob > 0:
        warnings.append(f"Trust out-of-bounds rows: {trust_oob}")
    if resolved_rate < cfg.warn_if_resolved_rate_lt:
        warnings.append(f"Resolved rate seems low ({pct(resolved_rate)} < {pct(cfg.warn_if_resolved_rate_lt)})")
    if resolved_rate > cfg.warn_if_resolved_rate_gt:
        warnings.append(f"Resolved rate seems high ({pct(resolved_rate)} > {pct(cfg.warn_if_resolved_rate_gt)})")
    if ticket_rate < cfg.warn_if_ticket_rate_lt:
        warnings.append(f"Ticket rate seems very low ({pct(ticket_rate)} < {pct(cfg.warn_if_ticket_rate_lt)})")
    if ticket_rate > cfg.warn_if_ticket_rate_gt:
        warnings.append(f"Ticket rate seems very high ({pct(ticket_rate)} > {pct(cfg.warn_if_ticket_rate_gt)})")
    if high_missing_cols:
        warnings.append(f"High-missingness columns (>= {pct(cfg.max_missing_frac_warn)}): {high_missing_cols[:20]}")

    # summary outputs
    summary = {
        "generated_at": now_ts,
        "run_id": run_id,
        "status": status,
        "repo_root": str(root),
        "input_csv": cfg.input_csv,
        "input_sha256": sha256_file(in_path),
        "rows": n,
        "columns": int(df.shape[1]),
        "columns_used": {
            "friction": friction_col,
            "trust": trust_col if has_trust else None,
            "ticket": ticket_col,
            "resolved": resolved_col,
            "repeat_7d": repeat7_col if has_repeat7 else None,
        },
        "coverage": {
            "critical_missing_frac": crit_missing,
            "missingness_top": {k: float(v) for k, v in missingness.head(15).items()},
        },
        "rates": {
            "resolved_rate": resolved_rate,
            "ticket_rate": ticket_rate,
            "repeat7_rate": repeat7_rate,
        },
        "correlations": corrs,
        "correlation_n": corr_n,
        "time_summary": time_summary,
        "warnings": warnings,
        "outputs": {
            "validation_report_md": str((out_dir / "validation_report.md").relative_to(root)),
            "validation_summary_json": str((out_dir / "validation_summary.json").relative_to(root)),
            "bucket_metrics_csv": str((out_dir / "validation_metrics_by_bucket.csv").relative_to(root)),
            "rep_metrics_csv": str((out_dir / "validation_metrics_by_rep.csv").relative_to(root)),
            "decile_lift_table_csv": str((out_dir / "decile_lift_table.csv").relative_to(root)),
            "logit_ticket_uncontrolled_csv": str((out_dir / "logit_ticket_uncontrolled_oddsratios.csv").relative_to(root)),
            "logit_ticket_controls_csv": str((out_dir / "logit_ticket_controls_oddsratios.csv").relative_to(root)),
            "logit_resolved_uncontrolled_csv": str((out_dir / "logit_resolved_uncontrolled_oddsratios.csv").relative_to(root)),
            "logit_resolved_controls_csv": str((out_dir / "logit_resolved_controls_oddsratios.csv").relative_to(root)),
            "model_summary_json": str((out_dir / "model_summary.json").relative_to(root)),
            "plots_run_dir": str(run_plots_dir.relative_to(root)),
        },
    }
    safe_write_json(out_dir / "validation_summary.json", summary)

    # report
    def fmt_corr(v: Optional[float], n_used: int) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return f"NA (n={n_used})"
        return f"{v:.3f} (n={n_used})"

    # pull ORs for friction_z for quick human read
    def or_line(df_or: pd.DataFrame) -> str:
        if df_or.empty or df_or.iloc[0].get("odds_ratio") is None:
            return "NA"
        r = df_or.iloc[0]
        return f"OR={r['odds_ratio']:.3f} (95% CI {r['ci_2p5']:.3f}–{r['ci_97p5']:.3f}) per +1 SD friction"

    report: List[str] = []
    report.append("# NovaFabric Validation Checklist\n")
    report.append(f"- Status: **{status}**")
    report.append(f"- Run ID: `{run_id}`")
    report.append(f"- Generated: {now_ts}")
    report.append(f"- Script: `{Path(__file__).resolve()}`")
    report.append(f"- Input: `{cfg.input_csv}`")
    report.append(f"- Input SHA256: `{sha256_file(in_path)}`")
    report.append(f"- Rows: **{n:,}** | Columns: **{df.shape[1]}**\n")

    report.append("## Key Rates")
    report.append(f"- Resolved rate: **{pct(resolved_rate)}**")
    report.append(f"- Ticket rate: **{pct(ticket_rate)}**")
    report.append(f"- Repeat-7D rate: **{pct(repeat7_rate)}**" if repeat7_rate is not None else "- Repeat-7D rate: NA")
    report.append("")

    report.append("## Directionality Checks (Correlations)")
    report.append(f"- friction vs trust: **{fmt_corr(corrs['corr_friction_trust'], corr_n['corr_friction_trust'])}** (expect negative)")
    report.append(f"- friction vs resolved: **{fmt_corr(corrs['corr_friction_resolved'], corr_n['corr_friction_resolved'])}** (often negative)")
    report.append(f"- friction vs ticket: **{fmt_corr(corrs['corr_friction_ticket'], corr_n['corr_friction_ticket'])}** (expect positive)")
    report.append(f"- friction vs repeat_7d: **{fmt_corr(corrs['corr_friction_repeat7'], corr_n['corr_friction_repeat7'])}** (often positive)")
    report.append("")

    report.append("## Evidence Pack A — Friction Decile Lift")
    report.append("Saved: `decile_lift_table.csv` and lift plots in the run plots folder.")
    report.append(df_to_markdown_safe(lift))
    report.append("")

    report.append("## Evidence Pack B — Logistic Regression (4 models)")
    report.append("Odds ratios are reported for **friction_z** (1 SD increase).")
    report.append(f"- Ticket (uncontrolled): {or_line(ticket_unctrl)}")
    report.append(f"- Ticket (+ subreason, rep controls): {or_line(ticket_ctrl)}")
    report.append(f"- Resolved (uncontrolled): {or_line(resolved_unctrl)}")
    report.append(f"- Resolved (+ subreason, rep controls): {or_line(resolved_ctrl)}")
    report.append("")
    report.append("Full metadata saved to `model_summary.json`.")
    report.append("")

    report.append("## Coverage Gate")
    report.append(f"- Fail threshold: missing fraction > {pct(cfg.fail_if_missing_frac_gt)}")
    for k, v in crit_missing.items():
        report.append(f"- {k}: missing {pct(v)}")
    if hard_fail_reasons:
        report.append(f"- ❌ FAIL reasons: {hard_fail_reasons}")
    else:
        report.append("- ✅ PASS")
    report.append("")

    report.append("## Warnings")
    if warnings:
        for w in warnings:
            report.append(f"- ⚠️ {w}")
    else:
        report.append("- None.")
    report.append("")

    report.append("## Plots")
    report.append(f"- Run plots folder: `{run_plots_dir.relative_to(root)}`")
    report.append("- hist_friction.png")
    if has_trust:
        report.append("- hist_trust.png")
    if not daily.empty:
        report.append("- ts_calls.png / ts_friction_mean.png / ts_ticket_rate.png / ts_resolved_rate.png")
    report.append("- lift_ticket_rate.png / lift_resolved_rate.png")
    if has_repeat7:
        report.append("- lift_repeat7_rate.png")
    report.append("")

    safe_write_text(out_dir / "validation_report.md", "\n".join(report))

    if cfg.hard_fail and status == "FAIL":
        raise ValueError(f"NovaFabric validation HARD FAIL: {hard_fail_reasons}")

    print("NovaFabric validation complete.")
    print(f"- Status: {status}")
    print(f"- Report : {out_dir / 'validation_report.md'}")
    print(f"- Summary: {out_dir / 'validation_summary.json'}")
    print(f"- Evidence: {out_dir / 'decile_lift_table.csv'} and logit_*_oddsratios.csv")
    print(f"- Plots  : {run_plots_dir}")


if __name__ == "__main__":
    main()
