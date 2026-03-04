#!/usr/bin/env python3
"""
02_evidence_pack.py

NovaFabric Validation Checklist (Evidence Pack)

Goal:
Create paper-proof artifacts beyond histograms/time series:
1) Versioned experiment lock (run folder + dataset hash + receipts)
2) Models with and without controls (ticket/resolved)
3) Negative control test (should show ~no effect)
4) Clean outputs for paper (CSVs + JSON summaries)

Reads:
- output/novafabric_validation/NovaFabric_instrumented.csv (default)
- output/novafabric_validation/instrumentation_receipt.json (optional; copied if exists)

Writes (per-run folder):
- output/novafabric_validation/evidence/run_<RUNID>/
    - evidence_summary.json
    - input_sha256.txt
    - copied_instrumentation_receipt.json (if present)
    - decile_lift_table.csv
    - lift_ticket_rate.png (optional)
    - lift_resolved_rate.png (optional)
    - lift_repeat7_rate.png (optional)
    - logit_ticket_uncontrolled_oddsratios.csv
    - logit_ticket_controls_oddsratios.csv
    - logit_resolved_uncontrolled_oddsratios.csv
    - logit_resolved_controls_oddsratios.csv
    - negative_control_results.csv
    - RUNSTAMP_<RUNID>.txt

Repo-root compliant: run from anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Repo-root helpers
# -----------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    return Path(__file__).resolve().parent.parent


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pick_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def to_int_flag(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s > 0.5).astype(int)


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    input_csv: str = "output/novafabric_validation/NovaFabric_instrumented.csv"
    out_dir: str = "output/novafabric_validation/evidence"

    # Which controls to include
    include_subreason: bool = True
    include_rep: bool = True

    # Reduce rep_id cardinality to avoid 800 dummy columns from nuking your laptop
    rep_min_calls: int = 50  # reps below this become "REP_OTHER"

    # Decile lift
    n_deciles: int = 10

    # Logistic regression settings
    max_iter: int = 2000
    C: float = 1.0
    bootstrap_n: int = 300
    random_seed: int = 42

    # Optional plots
    make_lift_plots: bool = True


# -----------------------------
# Modeling helpers
# -----------------------------
def build_design_matrix(
    df: pd.DataFrame,
    y_col: str,
    friction_col: str,
    subreason_col: Optional[str],
    rep_col: Optional[str],
    include_controls: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X, y for logistic regression.
    Always includes friction (standardized).
    Optionally adds one-hot controls for subreason and rep_id.
    """
    work = df.copy()

    # y
    y = to_int_flag(work[y_col]).values

    # friction numeric
    fr = pd.to_numeric(work[friction_col], errors="coerce")
    fr = fr.fillna(fr.median() if fr.notna().any() else 0.0).values.reshape(-1, 1)

    scaler = StandardScaler()
    fr_z = scaler.fit_transform(fr)  # standardize to make coefficients stable

    X_parts = [fr_z]
    names = ["friction_z"]

    if include_controls:
        if subreason_col and subreason_col in work.columns:
            sr = work[subreason_col].fillna("Other").astype(str)
            sr_d = pd.get_dummies(sr, prefix="subreason", drop_first=True)
            X_parts.append(sr_d.values)
            names.extend(sr_d.columns.tolist())

        if rep_col and rep_col in work.columns:
            rep = work[rep_col].fillna("REP_UNKNOWN").astype(str)
            rep_d = pd.get_dummies(rep, prefix="rep", drop_first=True)
            X_parts.append(rep_d.values)
            names.extend(rep_d.columns.tolist())

    X = np.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]
    return X, y, names


def fit_logit_oddsratios(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_iter: int,
    C: float,
    bootstrap_n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Fit logistic regression and return odds ratios + bootstrap CI.
    Uses sklearn LogisticRegression; bootstrap resamples rows.
    """
    rng = np.random.default_rng(seed)

    model = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
    model.fit(X, y)
    coef = model.coef_.ravel()

    # odds ratio per 1 unit increase in feature (friction_z is per 1 SD)
    or_hat = np.exp(coef)

    # bootstrap CI
    boot = np.zeros((bootstrap_n, X.shape[1]), dtype=float)
    n = len(y)

    for b in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        try:
            mb = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
            mb.fit(Xb, yb)
            boot[b, :] = np.exp(mb.coef_.ravel())
        except Exception:
            boot[b, :] = np.nan

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "odds_ratio": or_hat,
            "ci_2p5": lo,
            "ci_97p5": hi,
        }
    )

    # Focused "per +0.10 friction" interpretation:
    # friction_z is per SD; convert if we know SD of raw friction is approx:
    # We can’t infer raw SD here (X already standardized), so we report in SD units.
    # In your paper: “per 1 SD increase in friction, odds multiply by X.”
    # (If you want per 0.10 raw friction, we can add it later by passing raw SD.)
    return out.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# -----------------------------
# Lift table + plots
# -----------------------------
def decile_lift(
    df: pd.DataFrame,
    friction_col: str,
    ticket_col: str,
    resolved_col: str,
    repeat7_col: Optional[str],
    n_deciles: int = 10,
) -> pd.DataFrame:
    work = df.copy()
    fr = pd.to_numeric(work[friction_col], errors="coerce")
    work = work[fr.notna()].copy()
    work["friction"] = fr[fr.notna()].values

    work["ticket"] = to_int_flag(work[ticket_col]) if ticket_col in work.columns else 0
    work["resolved"] = to_int_flag(work[resolved_col]) if resolved_col in work.columns else 0
    if repeat7_col and repeat7_col in work.columns:
        work["repeat7"] = to_int_flag(work[repeat7_col])
    else:
        work["repeat7"] = np.nan

    # deciles (1..n_deciles)
    work["friction_decile"] = pd.qcut(work["friction"], q=n_deciles, labels=False, duplicates="drop") + 1

    g = work.groupby("friction_decile", as_index=False).agg(
        n=("friction", "size"),
        friction_mean=("friction", "mean"),
        ticket_rate=("ticket", "mean"),
        resolved_rate=("resolved", "mean"),
        repeat7_rate=("repeat7", "mean"),
    )

    # monotonicity checks (simple)
    def monotone_increasing(x: pd.Series) -> float:
        x = x.dropna().values
        if len(x) < 3:
            return float("nan")
        return float(np.mean(np.diff(x) >= 0))

    def monotone_decreasing(x: pd.Series) -> float:
        x = x.dropna().values
        if len(x) < 3:
            return float("nan")
        return float(np.mean(np.diff(x) <= 0))

    g["monotone_ticket_increasing_score"] = monotone_increasing(g["ticket_rate"])
    g["monotone_repeat7_increasing_score"] = monotone_increasing(g["repeat7_rate"])
    g["monotone_resolved_decreasing_score"] = monotone_decreasing(g["resolved_rate"])

    return g


def plot_lift(df_lift: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = df_lift[["friction_decile", y_col]].dropna()
    if d.empty:
        return
    plt.figure()
    plt.plot(d["friction_decile"], d[y_col])
    plt.title(title)
    plt.xlabel("friction decile (1=lowest)")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Negative control
# -----------------------------
def add_negative_control(df: pd.DataFrame, interaction_id_col: str = "interaction_id") -> pd.Series:
    """
    Deterministic pseudo-random variable derived from interaction_id.
    Should NOT predict ticket/resolved (if model is sane).
    """
    ids = df[interaction_id_col].astype(str).fillna("NA")
    # stable hashing -> [0,1)
    vals = ids.apply(lambda s: int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 2**32)
    return vals.astype(float)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = Config()
    root = find_repo_root()

    in_path = root / cfg.input_csv
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    # Versioned run folder
    run_id = uuid4().hex[:10]
    now_ts = datetime.now().isoformat(timespec="seconds")

    out_root = root / cfg.out_dir
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_write_text(run_dir / f"RUNSTAMP_{run_id}.txt", now_ts)

    # Lock dataset hash
    input_hash = sha256_file(in_path)
    safe_write_text(run_dir / "input_sha256.txt", input_hash)

    # Copy instrumentation receipt if available
    receipt_path = root / "output/novafabric_validation/instrumentation_receipt.json"
    receipt_copied = None
    if receipt_path.exists():
        receipt_copied = run_dir / "copied_instrumentation_receipt.json"
        receipt_copied.write_text(receipt_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("\n=== NovaFabric Evidence Pack ===")
    print("RUN_ID     :", run_id)
    print("NOW        :", now_ts)
    print("SCRIPT     :", Path(__file__).resolve())
    print("REPO_ROOT  :", root)
    print("INPUT_CSV  :", in_path)
    print("INPUT_SHA  :", input_hash[:16] + "...")
    print("RUN_DIR    :", run_dir)
    print("RECEIPT    :", receipt_path if receipt_path.exists() else "None found")
    print("================================\n")

    df = pd.read_csv(in_path, low_memory=False)

    # Prefer *_filled
    friction_col = pick_col(df, "friction_level_filled", "friction_level")
    ticket_col = pick_col(df, "ticket_flag_filled", "ticket_flag")
    resolved_col = pick_col(df, "resolved_flag_filled", "resolved_flag")
    repeat7_col = pick_col(df, "repeat_7d_flag_filled", "repeat_7d_flag")
    subreason_col = pick_col(df, "subreason_std_filled", "subreason_std")
    rep_col = pick_col(df, "rep_id_filled", "rep_id")

    # Coerce flags if present
    for c in [ticket_col, resolved_col, repeat7_col]:
        if c in df.columns:
            df[c] = to_int_flag(df[c])

    # Reduce rep_id cardinality (optional but recommended)
    if cfg.include_rep and rep_col in df.columns:
        rep_counts = df[rep_col].fillna("REP_UNKNOWN").astype(str).value_counts()
        keep = set(rep_counts[rep_counts >= cfg.rep_min_calls].index.tolist())
        df[rep_col] = df[rep_col].fillna("REP_UNKNOWN").astype(str).apply(lambda r: r if r in keep else "REP_OTHER")

    # ---- 1) Lift table + plots ----
    lift = decile_lift(
        df=df,
        friction_col=friction_col,
        ticket_col=ticket_col,
        resolved_col=resolved_col,
        repeat7_col=repeat7_col if repeat7_col in df.columns else None,
        n_deciles=cfg.n_deciles,
    )
    lift.to_csv(run_dir / "decile_lift_table.csv", index=False)

    if cfg.make_lift_plots:
        plot_lift(lift, "ticket_rate", "Ticket rate by friction decile", run_dir / "lift_ticket_rate.png")
        plot_lift(lift, "resolved_rate", "Resolved rate by friction decile", run_dir / "lift_resolved_rate.png")
        if "repeat7_rate" in lift.columns and lift["repeat7_rate"].notna().any():
            plot_lift(lift, "repeat7_rate", "Repeat-7D rate by friction decile", run_dir / "lift_repeat7_rate.png")

    # ---- 2) Logistic models: uncontrolled + controls ----
    # Ticket
    ticket_models = {}
    if ticket_col in df.columns:
        # uncontrolled
        X_u, y_u, names_u = build_design_matrix(
            df, y_col=ticket_col, friction_col=friction_col,
            subreason_col=None, rep_col=None,
            include_controls=False
        )
        or_ticket_u = fit_logit_oddsratios(X_u, y_u, names_u, cfg.max_iter, cfg.C, cfg.bootstrap_n, cfg.random_seed)
        or_ticket_u.to_csv(run_dir / "logit_ticket_uncontrolled_oddsratios.csv", index=False)
        ticket_models["uncontrolled"] = {"n": int(len(y_u)), "features": int(X_u.shape[1])}

        # controls
        X_c, y_c, names_c = build_design_matrix(
            df, y_col=ticket_col, friction_col=friction_col,
            subreason_col=subreason_col if cfg.include_subreason else None,
            rep_col=rep_col if cfg.include_rep else None,
            include_controls=True
        )
        or_ticket_c = fit_logit_oddsratios(X_c, y_c, names_c, cfg.max_iter, cfg.C, cfg.bootstrap_n, cfg.random_seed)
        or_ticket_c.to_csv(run_dir / "logit_ticket_controls_oddsratios.csv", index=False)
        ticket_models["controls"] = {"n": int(len(y_c)), "features": int(X_c.shape[1])}

    # Resolved
    resolved_models = {}
    if resolved_col in df.columns:
        # uncontrolled
        X_u, y_u, names_u = build_design_matrix(
            df, y_col=resolved_col, friction_col=friction_col,
            subreason_col=None, rep_col=None,
            include_controls=False
        )
        or_res_u = fit_logit_oddsratios(X_u, y_u, names_u, cfg.max_iter, cfg.C, cfg.bootstrap_n, cfg.random_seed)
        or_res_u.to_csv(run_dir / "logit_resolved_uncontrolled_oddsratios.csv", index=False)
        resolved_models["uncontrolled"] = {"n": int(len(y_u)), "features": int(X_u.shape[1])}

        # controls
        X_c, y_c, names_c = build_design_matrix(
            df, y_col=resolved_col, friction_col=friction_col,
            subreason_col=subreason_col if cfg.include_subreason else None,
            rep_col=rep_col if cfg.include_rep else None,
            include_controls=True
        )
        or_res_c = fit_logit_oddsratios(X_c, y_c, names_c, cfg.max_iter, cfg.C, cfg.bootstrap_n, cfg.random_seed)
        or_res_c.to_csv(run_dir / "logit_resolved_controls_oddsratios.csv", index=False)
        resolved_models["controls"] = {"n": int(len(y_c)), "features": int(X_c.shape[1])}

    # ---- 3) Negative control test ----
    # Create a "should-not-matter" variable and see if it predicts ticket/resolved.
    neg = add_negative_control(df, interaction_id_col="interaction_id" if "interaction_id" in df.columns else df.columns[0])
    df["neg_control"] = neg

    neg_results = []
    for target in [ticket_col, resolved_col]:
        if target not in df.columns:
            continue

        # Model: target ~ neg_control (+ same controls)
        work = df.copy()
        work["neg_control"] = pd.to_numeric(work["neg_control"], errors="coerce").fillna(work["neg_control"].median())

        # Build X: neg_control + controls (but NOT friction)
        y = to_int_flag(work[target]).values
        neg_x = work["neg_control"].values.reshape(-1, 1)
        neg_x = StandardScaler().fit_transform(neg_x)

        X_parts = [neg_x]
        names = ["neg_control_z"]

        if cfg.include_subreason and subreason_col in work.columns:
            sr = work[subreason_col].fillna("Other").astype(str)
            sr_d = pd.get_dummies(sr, prefix="subreason", drop_first=True)
            X_parts.append(sr_d.values)
            names.extend(sr_d.columns.tolist())

        if cfg.include_rep and rep_col in work.columns:
            rep = work[rep_col].fillna("REP_UNKNOWN").astype(str)
            rep_d = pd.get_dummies(rep, prefix="rep", drop_first=True)
            X_parts.append(rep_d.values)
            names.extend(rep_d.columns.tolist())

        X = np.concatenate(X_parts, axis=1)

        or_df = fit_logit_oddsratios(X, y, names, cfg.max_iter, cfg.C, cfg.bootstrap_n, cfg.random_seed)
        # keep only the neg term for the paper sanity check
        row = or_df[or_df["feature"] == "neg_control_z"].iloc[0].to_dict()
        row["target"] = target
        neg_results.append(row)

    neg_out = pd.DataFrame(neg_results)
    neg_out.to_csv(run_dir / "negative_control_results.csv", index=False)

    # ---- 4) Evidence summary JSON ----
    summary = {
        "generated_at": now_ts,
        "run_id": run_id,
        "input_csv": str(in_path.relative_to(root)),
        "input_sha256": input_hash,
        "receipt_copied": str(receipt_copied.relative_to(root)) if receipt_copied else None,
        "columns_used": {
            "friction": friction_col,
            "ticket": ticket_col if ticket_col in df.columns else None,
            "resolved": resolved_col if resolved_col in df.columns else None,
            "repeat7": repeat7_col if repeat7_col in df.columns else None,
            "subreason": subreason_col if subreason_col in df.columns else None,
            "rep": rep_col if rep_col in df.columns else None,
        },
        "lift_monotonicity": {
            "ticket_increasing_score": float(lift["monotone_ticket_increasing_score"].iloc[0]) if "monotone_ticket_increasing_score" in lift.columns else None,
            "repeat7_increasing_score": float(lift["monotone_repeat7_increasing_score"].iloc[0]) if "monotone_repeat7_increasing_score" in lift.columns else None,
            "resolved_decreasing_score": float(lift["monotone_resolved_decreasing_score"].iloc[0]) if "monotone_resolved_decreasing_score" in lift.columns else None,
        },
        "models": {
            "ticket": ticket_models,
            "resolved": resolved_models,
        },
        "outputs": {
            "run_dir": str(run_dir.relative_to(root)),
            "decile_lift_table": str((run_dir / "decile_lift_table.csv").relative_to(root)),
            "neg_control_results": str((run_dir / "negative_control_results.csv").relative_to(root)),
        },
        "notes": [
            "Logistic ORs are per 1 SD increase in friction (friction_z).",
            "Negative control should have OR ~ 1.0 with CI spanning 1.0; if not, revisit modeling leakage.",
            "Rep_id was collapsed: reps with < rep_min_calls -> REP_OTHER to control dimensionality.",
        ],
        "config": cfg.__dict__,
    }
    safe_write_json(run_dir / "evidence_summary.json", summary)

    print("Evidence pack complete.")
    print("Run folder:", run_dir)
    print("Key outputs:")
    print("-", run_dir / "decile_lift_table.csv")
    print("-", run_dir / "logit_ticket_uncontrolled_oddsratios.csv")
    print("-", run_dir / "logit_ticket_controls_oddsratios.csv")
    print("-", run_dir / "logit_resolved_uncontrolled_oddsratios.csv")
    print("-", run_dir / "logit_resolved_controls_oddsratios.csv")
    print("-", run_dir / "negative_control_results.csv")
    print("-", run_dir / "evidence_summary.json")


if __name__ == "__main__":
    main()
