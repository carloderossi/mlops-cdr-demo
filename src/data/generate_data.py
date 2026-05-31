"""
generate_data.py
----------------
Generates synthetic credit risk datasets for the Champion–Challenger MLOps demo.

Produces three datasets:
  - baseline_train.parquet   : stable economic conditions, used to train Champion (ModelA)
  - baseline_test.parquet    : held-out test split from the same stable distribution
  - drifted_inference.parquet: simulated economic downturn (covariate drift)
  - drifted_train.parquet    : retrain mix — historical stable + drifted, used for Challenger (ModelB)

Drift simulation follows the spec:
  - lower average income          (income mean -18%)
  - higher debt-to-income ratio   (+22%)
  - higher default probability    (via shifted feature distributions)
  - reduced financial stability

Usage:
    python generate_data.py [--output_dir ./] [--seed 42]
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_credit_dataset(
    n_samples: int,
    income_mean: float,
    income_std: float,
    age_mean: float,
    age_std: float,
    debt_ratio_mean: float,
    debt_ratio_std: float,
    employment_years_mean: float,
    credit_score_mean: float,
    credit_score_std: float,
    base_default_rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic tabular credit risk dataset.

    Features:
        income              : annual gross income (CHF/EUR equivalent)
        age                 : customer age in years
        debt_ratio          : total debt / annual income
        num_credit_lines    : number of open credit lines
        employment_years    : years at current employer
        credit_score        : simplified internal score (300–850)
        num_late_payments   : late payments in past 24 months
        loan_amount         : requested loan amount
        loan_to_income      : loan_amount / income (derived)

    Target:
        default             : 1 = default, 0 = no default (binary)
    """
    rng = np.random.default_rng(seed)

    income = rng.normal(income_mean, income_std, n_samples).clip(15_000, 500_000)
    age = rng.normal(age_mean, age_std, n_samples).clip(18, 75).astype(int)
    debt_ratio = rng.normal(debt_ratio_mean, debt_ratio_std, n_samples).clip(0.0, 2.0)
    num_credit_lines = rng.integers(1, 15, n_samples)
    employment_years = rng.exponential(employment_years_mean, n_samples).clip(0, 40)
    credit_score = rng.normal(credit_score_mean, credit_score_std, n_samples).clip(300, 850).astype(int)
    num_late_payments = rng.integers(0, 12, n_samples)
    loan_amount = (income * rng.uniform(0.5, 3.0, n_samples)).clip(5_000, 300_000)
    loan_to_income = loan_amount / income

    # Default probability: logistic function of key risk drivers
    log_odds = (
        -3.5
        + 1.8  * debt_ratio
        - 0.003 * (credit_score - 600)
        + 0.4  * num_late_payments
        + 0.5  * loan_to_income
        - 0.015 * employment_years
        - 0.0000015 * income
        + rng.normal(0, 0.3, n_samples)   # noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))

    # Scale to target base default rate
    prob_default = prob_default * (base_default_rate / prob_default.mean())
    prob_default = prob_default.clip(0.0, 1.0)

    default = (rng.uniform(0, 1, n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "income": income.round(2),
        "age": age,
        "debt_ratio": debt_ratio.round(4),
        "num_credit_lines": num_credit_lines,
        "employment_years": employment_years.round(2),
        "credit_score": credit_score,
        "num_late_payments": num_late_payments,
        "loan_amount": loan_amount.round(2),
        "loan_to_income": loan_to_income.round(4),
        "default": default,
    })

    return df


# ---------------------------------------------------------------------------
# Regime configurations
# ---------------------------------------------------------------------------

STABLE_PARAMS = dict(
    income_mean=62_000,
    income_std=18_000,
    age_mean=42,
    age_std=12,
    debt_ratio_mean=0.32,
    debt_ratio_std=0.15,
    employment_years_mean=7.0,
    credit_score_mean=660,
    credit_score_std=80,
    base_default_rate=0.18,   # ~18% default rate in stable conditions
)

# Economic downturn: -18% income, +22% debt ratio, higher default rate
DOWNTURN_PARAMS = dict(
    income_mean=62_000 * 0.82,   # -18%
    income_std=20_000,
    age_mean=40,
    age_std=13,
    debt_ratio_mean=0.32 * 1.22,  # +22%
    debt_ratio_std=0.20,
    employment_years_mean=5.0,    # shorter tenure due to layoffs
    credit_score_mean=630,        # slightly worse scores
    credit_score_std=90,
    base_default_rate=0.27,       # higher default rate ~+9pp
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str = "./data", seed: int = 42):
    print("Output dir:", output_dir)
    print("Seed:", seed)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Baseline: train + test split (Champion ModelA training data)
    print("[INFO] Generating baseline dataset (stable economy)...")
    baseline_full = generate_credit_dataset(n_samples=8_000, seed=seed, **STABLE_PARAMS)
    baseline_train, baseline_test = train_test_split(
        baseline_full, test_size=0.2, random_state=seed, stratify=baseline_full["default"]
    )
    baseline_train = baseline_train.reset_index(drop=True)
    baseline_test = baseline_test.reset_index(drop=True)

    baseline_train.to_parquet(os.path.join(output_dir, "baseline_train.parquet"), index=False)
    baseline_test.to_parquet(os.path.join(output_dir, "baseline_test.parquet"), index=False)

    print(f"  baseline_train : {len(baseline_train):,} rows | default rate: {baseline_train['default'].mean():.2%}")
    print(f"  baseline_test  : {len(baseline_test):,} rows  | default rate: {baseline_test['default'].mean():.2%}")

    # 2. Drifted inference data (no labels exposed — simulates production traffic)
    print("[INFO] Generating drifted inference dataset (economic downturn)...")
    drifted_full = generate_credit_dataset(n_samples=3_000, seed=seed + 10, **DOWNTURN_PARAMS)

    # Inference data ships without labels (label col dropped)
    drifted_inference = drifted_full.drop(columns=["default"]).reset_index(drop=True)
    drifted_labels = drifted_full[["default"]].reset_index(drop=True)  # "delayed ground truth"

    drifted_inference.to_parquet(os.path.join(output_dir, "drifted_inference.parquet"), index=False)
    drifted_labels.to_parquet(os.path.join(output_dir, "drifted_labels.parquet"), index=False)

    print(f"  drifted_inference: {len(drifted_inference):,} rows | labels withheld (delayed ground truth)")
    print(f"  drifted_labels   : {len(drifted_labels):,} rows | revealed after delay")

    # 3. Challenger retraining mix: 70% stable history + 100% drifted labelled
    print("[INFO] Generating challenger retrain dataset...")
    stable_sample = baseline_train.sample(frac=0.7, random_state=seed).reset_index(drop=True)
    drifted_labelled = drifted_full.reset_index(drop=True)
    drifted_train = pd.concat([stable_sample, drifted_labelled], ignore_index=True).sample(
        frac=1, random_state=seed
    )
    drifted_train.to_parquet(os.path.join(output_dir, "drifted_train.parquet"), index=False)

    print(f"  drifted_train : {len(drifted_train):,} rows | default rate: {drifted_train['default'].mean():.2%}")
    print(f"    ↳ {len(stable_sample):,} stable rows + {len(drifted_labelled):,} drifted rows")

    # 4. Summary stats for drift explanation panel
    print("\n[INFO] Drift explanation summary:")
    for col in ["income", "debt_ratio", "credit_score"]:
        b_mean = baseline_full[col].mean()
        d_mean = drifted_full[col].mean()
        delta_pct = (d_mean - b_mean) / b_mean * 100
        print(f"  {col:<22}: baseline={b_mean:.2f}  drifted={d_mean:.2f}  Δ={delta_pct:+.1f}%")

    b_dr = baseline_full["default"].mean()
    d_dr = drifted_full["default"].mean()
    print(f"  {'default_rate':<22}: baseline={b_dr:.2%}  drifted={d_dr:.2%}  Δ={((d_dr-b_dr)/b_dr*100):+.1f}%")

    print(f"\n[OK] All datasets written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit risk datasets")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.output_dir, args.seed)
