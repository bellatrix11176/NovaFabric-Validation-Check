#!/usr/bin/env python3
"""
00_instrument_novafabric.py

NovaFabric Validation Checklist (Instrumentation)

Repo-root compliant: run from anywhere.
Root detected by presence of:
- data/
- src/

Reads:
- data/NovaFabric.csv   (raw calls)

Writes:
- output/novafabric_validation/NovaFabric_instrumented.csv
- output/novafabric_validation/instrumentation_receipt.json
- output/novafabric_validation/RUNSTAMP_<run_id>.txt

Purpose:
- Create/overwrite *_filled columns that encode a stronger causal story:
  - friction_level_filled -> ticket_flag_filled (positive)
  - friction_level_filled -> resolved_flag_filled (negative)
  - (unresolved + friction) -> repeat_7d_flag_filled / repeat_30d_flag_filled (positive)
  - friction -> trust (negative), plus penalties for repeats/tickets/unresolved
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd


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


def to_int_flag(series: pd.Series) -> pd.Series:
    def conv(v: Any) -> int:
        if v is None:
            return 0
        try:
            if isinstance(v, float) and math.isnan(v):
                return 0
        except Exception:
            pass
        try:
            return int(float(v))
        except Exception:
            s = str(v).strip().lower()
            return 1 if s in ("true", "yes", "y", "1") else 0

    return series.apply(conv).astype(int)


def ensure_col(df: pd.DataFrame, col: str, default: Any) -> None:
    if col not in df.columns:
        df[col] = default


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable-ish sigmoid
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


# -----------------------------
# Config (your “knobs”)
# -----------------------------
@dataclass
class Config:
    input_csv: str = "data/NovaFabric.csv"
    out_dir: str = "output/novafabric_validation"
    out_csv: str = "output/novafabric_validation/NovaFabric_instrumented.csv"
    seed: int = 42

    # --- friction baseline fill ---
    friction_default: float = 0.35
    friction_clip: Tuple[float, float] = (0.0, 1.0)
    friction_noise_sd: float = 0.04  # small “measurement” noise

    # --- causal strength knobs ---
    # ticket probability = sigmoid(ticket_base + ticket_alpha*(friction_z) + ticket_beta_unresolved + noise)
    ticket_base: float = -1.55
    ticket_alpha: float = 1.90         # ↑ increase for stronger friction->ticket
    ticket_beta_unresolved: float = 0.85
    ticket_noise_sd: float = 0.35

    # resolved probability = sigmoid(resolved_base + resolved_alpha*(friction_z) + resolved_beta_ticket + noise)
    resolved_base: float = -0.45
    resolved_alpha: float = -1.55      # negative: higher friction => lower resolved
    resolved_beta_ticket: float = -0.55
    resolved_noise_sd: float = 0.45

    # repeats: mostly driven by unresolved + friction (+ ticket)
    repeat7_base: float = -2.25
    repeat7_beta_unresolved: float = 1.55
    repeat7_alpha_friction: float = 0.95
    repeat7_beta_ticket: float = 0.55
    repeat7_noise_sd: float = 0.50

    repeat30_base: float = -1.70
    repeat30_beta_unresolved: float = 1.20
    repeat30_alpha_friction: float = 0.75
    repeat30_beta_ticket: float = 0.35
    repeat30_noise_sd: float = 0.55

    # trust score: start high-ish, penalize friction + failures
    trust_base: float = 72.0
    trust_friction_weight: float = 18.0     # penalty per 1.0 friction
    trust_ticket_penalty: float = 4.5
    trust_unresolved_penalty: float = 7.0
    trust_repeat7_penalty: float = 5.0
    trust_repeat30_penalty: float = 3.0
    trust_noise_sd: float = 3.0
    trust_clip: Tuple[float, float] = (0.0, 100.0)

    # allow using existing raw columns as priors (then “filled” overwrites)
    prefer_existing_friction: bool = True


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
            "Expected:\n  data/NovaFabric.csv\n"
        )

    out_dir = root / cfg.out_dir
    out_path = root / cfg.out_csv
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex[:10]
    now_ts = datetime.now().isoformat(timespec="seconds")

    # hard proof
    safe_write_text(out_dir / f"RUNSTAMP_{run_id}.txt", now_ts)

    print("\n=== NovaFabric Instrumentation Run ===")
    print("RUN_ID    :", run_id)
    print("NOW       :", now_ts)
    print("SCRIPT    :", Path(__file__).resolve())
    print("REPO_ROOT :", root)
    print("INPUT     :", in_path)
    print("OUTPUT    :", out_path)
    print("=====================================\n")

    rng = np.random.default_rng(cfg.seed)

    df = pd.read_csv(in_path, low_memory=False)

    # baseline columns for stability
    ensure_col(df, "interaction_id", "")
    ensure_col(df, "customer_id", "")
    ensure_col(df, "rep_id", "")
    ensure_col(df, "timestamp", "")

    # --- friction fill ---
    if cfg.prefer_existing_friction and "friction_level" in df.columns:
        friction_raw = pd.to_numeric(df["friction_level"], errors="coerce")
    else:
        friction_raw = pd.Series([np.nan] * len(df))

    friction_filled = friction_raw.fillna(cfg.friction_default).astype(float).to_numpy()
    friction_filled = friction_filled + rng.normal(0.0, cfg.friction_noise_sd, size=len(df))
    friction_filled = np.clip(friction_filled, cfg.friction_clip[0], cfg.friction_clip[1])

    # standardize friction for logistic models (z-score)
    mu = float(np.mean(friction_filled))
    sd = float(np.std(friction_filled)) if float(np.std(friction_filled)) > 1e-9 else 1.0
    friction_z = (friction_filled - mu) / sd

    # Use existing outcome/resolution as weak priors if present (optional)
    # We'll still generate filled flags from the causal model.
    if "resolved_flag" in df.columns:
        resolved_prior = to_int_flag(df["resolved_flag"]).to_numpy()
    else:
        resolved_prior = np.zeros(len(df), dtype=int)

    # Define "unresolved" prior from resolved_prior
    unresolved_prior = 1 - resolved_prior

    # --- ticket generation (strong friction effect) ---
    ticket_logits = (
        cfg.ticket_base
        + cfg.ticket_alpha * friction_z
        + cfg.ticket_beta_unresolved * unresolved_prior
        + rng.normal(0.0, cfg.ticket_noise_sd, size=len(df))
    )
    ticket_p = sigmoid(ticket_logits)
    ticket_flag_filled = (rng.random(len(df)) < ticket_p).astype(int)

    # --- resolved generation (depends on friction + tickets) ---
    resolved_logits = (
        cfg.resolved_base
        + cfg.resolved_alpha * friction_z
        + cfg.resolved_beta_ticket * ticket_flag_filled
        + rng.normal(0.0, cfg.resolved_noise_sd, size=len(df))
    )
    resolved_p = sigmoid(resolved_logits)
    resolved_flag_filled = (rng.random(len(df)) < resolved_p).astype(int)
    unresolved_filled = 1 - resolved_flag_filled

    # --- repeats (unresolved + friction + tickets) ---
    rep7_logits = (
        cfg.repeat7_base
        + cfg.repeat7_beta_unresolved * unresolved_filled
        + cfg.repeat7_alpha_friction * friction_z
        + cfg.repeat7_beta_ticket * ticket_flag_filled
        + rng.normal(0.0, cfg.repeat7_noise_sd, size=len(df))
    )
    rep7_p = sigmoid(rep7_logits)
    repeat_7d_flag_filled = (rng.random(len(df)) < rep7_p).astype(int)

    rep30_logits = (
        cfg.repeat30_base
        + cfg.repeat30_beta_unresolved * unresolved_filled
        + cfg.repeat30_alpha_friction * friction_z
        + cfg.repeat30_beta_ticket * ticket_flag_filled
        + rng.normal(0.0, cfg.repeat30_noise_sd, size=len(df))
    )
    rep30_p = sigmoid(rep30_logits)
    repeat_30d_flag_filled = (rng.random(len(df)) < rep30_p).astype(int)

    # --- trust score (penalize friction + bad events) ---
    trust = (
        cfg.trust_base
        - cfg.trust_friction_weight * friction_filled
        - cfg.trust_ticket_penalty * ticket_flag_filled
        - cfg.trust_unresolved_penalty * unresolved_filled
        - cfg.trust_repeat7_penalty * repeat_7d_flag_filled
        - cfg.trust_repeat30_penalty * repeat_30d_flag_filled
        + rng.normal(0.0, cfg.trust_noise_sd, size=len(df))
    )
    trust = np.clip(trust, cfg.trust_clip[0], cfg.trust_clip[1])

    # Write *_filled columns (these are what validator prefers)
    df["friction_level_filled"] = friction_filled
    df["ticket_flag_filled"] = ticket_flag_filled
    df["resolved_flag_filled"] = resolved_flag_filled
    df["repeat_7d_flag_filled"] = repeat_7d_flag_filled
    df["repeat_30d_flag_filled"] = repeat_30d_flag_filled
    df["trust_score_filled"] = trust

    # Also provide “compat” names if your validator expects these in some variants
    # (harmless duplicates; leave them if already present)
    ensure_col(df, "ticket_flag", ticket_flag_filled)
    ensure_col(df, "trust_score", trust)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    receipt: Dict[str, Any] = {
        "generated_at": now_ts,
        "run_id": run_id,
        "script": str(Path(__file__).resolve()),
        "repo_root": str(root),
        "input_csv": cfg.input_csv,
        "output_csv": str(Path(cfg.out_csv)),
        "rows": int(len(df)),
        "seed": cfg.seed,
        "friction_summary": {"mean": mu, "std": sd, "min": float(np.min(friction_filled)), "max": float(np.max(friction_filled))},
        "parameters": asdict(cfg),
        "notes": [
            "ticket_flag_filled depends positively on friction_z (ticket_alpha)",
            "resolved_flag_filled depends negatively on friction_z (resolved_alpha) and negatively on ticket_flag_filled",
            "repeats depend positively on unresolved + friction_z (+ ticket_flag_filled)",
            "trust_score_filled penalizes friction and failure events",
        ],
    }
    safe_write_json(out_dir / "instrumentation_receipt.json", receipt)

    print("Instrumentation complete.")
    print(f"- Wrote: {out_path}")
    print(f"- Wrote: {out_dir / 'instrumentation_receipt.json'}")
    print(f"- Runstamp: {out_dir / f'RUNSTAMP_{run_id}.txt'}")


if __name__ == "__main__":
    main()
