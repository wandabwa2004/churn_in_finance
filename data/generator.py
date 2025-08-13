# generator.py
# One file with small, readable functions. Each stage is bite-sized and testable.

from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd

from config import Config

# ---------------------------
# Helpers
# ---------------------------

def rng_from_seed(seed: int) -> np.random.Generator:
    """Single RNG so runs are reproducible."""
    return np.random.default_rng(seed)

def as_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def rand_dates_uniform(rng: np.random.Generator, n: int, start: datetime, end: datetime) -> np.ndarray:
    """Uniform random dates in [start, end)."""
    span = (end - start).days
    offsets = rng.integers(low=0, high=span, size=n)
    return np.array([start + timedelta(days=int(d)) for d in offsets])

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def rnb(rng: np.random.Generator, mu: np.ndarray, r: float) -> np.ndarray:
    """
    Negative Binomial with mean=mu and dispersion r.
    Var = mu + mu^2 / r.
    We convert to numpy's (n, p): n=r, p=r/(r+mu).
    """
    mu = np.clip(mu, 1e-9, None)
    p = r / (r + mu)
    return rng.negative_binomial(n=r, p=p, size=mu.shape[0])

def rlognormal_from_median(rng: np.random.Generator, median: np.ndarray, sigma: float) -> np.ndarray:
    """
    Lognormal parameterized by median. median = exp(mu).
    sigma controls right-skew (0.4..0.8 feels realistic for money).
    """
    mu = np.log(np.clip(median, 1e-9, None))
    return rng.lognormal(mean=mu, sigma=sigma, size=median.shape[0])

def rgamma_from_mean_k(rng: np.random.Generator, mean: np.ndarray, k: float) -> np.ndarray:
    """Gamma with shape k and scale theta=mean/k."""
    mean = np.clip(mean, 1e-9, None)
    theta = mean / k
    return rng.gamma(shape=k, scale=theta, size=mean.shape[0])

def zero_inflate(values: np.ndarray, mask_zero: np.ndarray) -> np.ndarray:
    """Force zeros where a channel is off (e.g., app not used)."""
    out = values.copy()
    out[mask_zero] = 0
    return out

# ---------------------------
# Stage 1: Base population
# ---------------------------

def generate_population(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Regions, urban vs rural, demographics, account open date."""
    regions = np.array(list(cfg.region_weights.keys()))
    region_p = np.array(list(cfg.region_weights.values()))
    region = rng.choice(regions, size=cfg.n, p=region_p)

    # Urban share depends on region (Nairobi ~ fully urban)
    urban_p = np.vectorize(cfg.urban_share.get)(region)
    urban = (rng.random(cfg.n) < urban_p).astype(int)

    # Demographics — realistic-ish but simple
    # Age: bias to working age via Beta on [18, 70]
    age = (18 + (70 - 18) * rng.beta(a=2.0, b=2.5, size=cfg.n)).astype(int)
    gender = rng.choice(["Male", "Female"], size=cfg.n, p=[0.49, 0.51])
    education = rng.choice(["None", "Primary", "Secondary", "Tertiary"],
                           size=cfg.n, p=[0.05, 0.25, 0.45, 0.25])
    employment = rng.choice(["Employed", "Self-employed", "Informal", "Unemployed"],
                            size=cfg.n, p=[0.45, 0.20, 0.25, 0.10])
    kyc = rng.choice(["Yes", "No"], size=cfg.n, p=[0.92, 0.08])

    # Uniform arrival of customers across the window
    start, end = as_dt(cfg.start), as_dt(cfg.end)
    acct_open = rand_dates_uniform(rng, cfg.n, start, end)

    return pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(cfg.n)],
        "account_open_date": acct_open,
        "age": age,
        "gender": gender,
        "region": region,
        "urban": urban,
        "education_level": education,
        "employment_status": employment,
        "KYC_verified": kyc,
    })

# ---------------------------
# Stage 2: Adoption flags
# ---------------------------

def generate_adoption(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Flags for Equity app, Equitel, M-Pesa linkage."""
    urban = df["urban"].values
    edu = df["education_level"].values
    emp = df["employment_status"].values

    # App usage — more likely in urban, tertiary, and (self-)employed
    base_app = 0.50 + 0.30 * urban
    base_app += 0.05 * (edu == "Tertiary")
    base_app += 0.05 * np.isin(emp, ["Employed", "Self-employed"])
    uses_app = (rng.random(df.shape[0]) < np.clip(base_app, 0.05, 0.98)).astype(int)

    # Equitel: modest adoption, slightly higher in urban
    uses_equitel = (rng.random(df.shape[0]) < (0.25 + 0.10 * urban)).astype(int)

    # M-Pesa linked to bank: very common, esp. urban
    mpesa_linked = (rng.random(df.shape[0]) < (0.75 + 0.15 * urban)).astype(int)

    df = df.copy()
    df["uses_equity_mobile_app"] = uses_app
    df["uses_equitel"] = uses_equitel
    df["mpesa_linked_to_bank"] = mpesa_linked
    return df

# ---------------------------
# Stage 3: Transaction & activity counts
# ---------------------------

def generate_counts(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Overdispersed counts with zero-inflation when channels are off."""
    urban = df["urban"].values
    uses_app = df["uses_equity_mobile_app"].values
    uses_equitel = df["uses_equitel"].values
    mpesa_linked = df["mpesa_linked_to_bank"].values

    # App sessions last 30d: zero if no app, otherwise NB
    mu_sessions = 4 + 4 * urban
    sessions = rnb(rng, mu_sessions, r=cfg.nb_r)
    sessions = zero_inflate(sessions, mask_zero=(uses_app == 0))

    # Mobile txn count last 90d: bigger for urban/app/equitel
    mu_mob_txn = 6 + 6 * urban + 4 * uses_app + 2 * uses_equitel
    mob_txn = rnb(rng, mu_mob_txn, r=cfg.nb_r)

    # M-Pesa cash in/out (counts): strong uplift if linked
    mu_in = (5 + 4 * urban) * np.where(mpesa_linked == 1, 1.0, 0.3)
    mu_out = (4 + 3 * urban) * np.where(mpesa_linked == 1, 1.0, 0.3)
    mpesa_in = rnb(rng, mu_in, r=cfg.nb_r)
    mpesa_out = rnb(rng, mu_out, r=cfg.nb_r)

    # Deposits / withdrawals (all channels)
    mu_dep = 3 + 1 * urban + 0.5 * uses_app
    mu_wdr = 3 + 1 * urban
    dep = rnb(rng, mu_dep, r=3.0)
    wdr = rnb(rng, mu_wdr, r=3.0)

    # Branch visits (less for urban), complaints (slightly more for urban due to volume)
    branch = np.clip(rng.poisson(lam=(2.2 - 1.0 * urban), size=df.shape[0]), a_min=0, a_max=None)
    complaints = rng.poisson(lam=(0.15 + 0.10 * urban), size=df.shape[0])

    df = df.copy()
    df["equity_mobile_sessions_last_30d"] = sessions
    df["equity_mobile_txn_count_last_90d"] = mob_txn
    df["mpesa_cash_in_count"] = mpesa_in
    df["mpesa_cash_out_count"] = mpesa_out
    df["num_deposits_last_90d"] = dep
    df["num_withdrawals_last_90d"] = wdr
    df["branch_visits_count_last_year"] = branch
    df["complaints_count_last_year"] = complaints
    return df

# ---------------------------
# Stage 4: Monetary amounts (skewed)
# ---------------------------

def generate_amounts(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Lognormal / Gamma draws for realistic right-skewed money amounts."""
    urban = df["urban"].values
    emp = df["employment_status"].values

    # Transaction volume (90d): higher median for urban
    median_vol = np.where(urban == 1, cfg.median_vol_urban, cfg.median_vol_rural)
    vol_90d = rlognormal_from_median(rng, median_vol, sigma=cfg.sigma_vol)

    # Average balance (6m): higher for urban + uplift for (self-)employed
    median_bal = np.where(urban == 1, cfg.median_bal_urban, cfg.median_bal_rural)
    median_bal = median_bal + 4_000.0 * np.isin(emp, ["Employed", "Self-employed"])
    bal_6m = rlognormal_from_median(rng, median_bal, sigma=cfg.sigma_bal)

    # Fuliza overdraft (Gamma): slightly higher reliance rural
    fuliza_mean = cfg.fuliza_mean_base + 600.0 * (1 - urban)
    fuliza_amt = rgamma_from_mean_k(rng, fuliza_mean, k=2.0)

    # M-Shwari savings (Gamma with many zeros)
    is_zero = rng.random(df.shape[0]) < cfg.mshwari_zero_prob
    mshwari_mean = cfg.mshwari_mean_base + 2_000.0 * urban
    mshwari = np.where(is_zero, 0.0, rgamma_from_mean_k(rng, mshwari_mean, k=2.0))

    df = df.copy()
    df["equity_mobile_trans_volume_last_90d"] = np.round(vol_90d, 2)
    df["avg_balance_last_6m"] = np.round(bal_6m, 2)
    df["fuliza_overdraft_amt"] = np.round(fuliza_amt, 2)
    df["mshwari_savings_balance"] = np.round(mshwari, 2)
    return df

# ---------------------------
# Stage 5: Loans & repayment rate
# ---------------------------

def generate_credit(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Loan applications and repayment quality (Beta on [0,1])."""
    urban = df["urban"].values
    mpesa_linked = df["mpesa_linked_to_bank"].values

    # Loan applications — modest uplift for urban
    loan_apps = rng.poisson(lam=(1.1 + 0.2 * urban), size=df.shape[0])

    # M-Shwari loans — more if linked
    mshwari_loans = rng.poisson(lam=(1.0 + 0.8 * mpesa_linked), size=df.shape[0])

    # Repayment rate depends on employment segment
    emp = df["employment_status"].values
    alpha_beta = {
        "Employed":      (6.0, 1.5),
        "Self-employed": (5.0, 1.8),
        "Informal":      (4.5, 2.0),
        "Unemployed":    (3.8, 2.4),
    }
    a = np.array([alpha_beta[e][0] for e in emp])
    b = np.array([alpha_beta[e][1] for e in emp])
    repay_rate = rng.beta(a, b)

    df = df.copy()
    df["loan_applications_count"] = loan_apps
    df["mshwari_loans_count"] = mshwari_loans
    df["loan_repayment_rate"] = np.round(repay_rate, 4)
    return df

# ---------------------------
# Stage 6: Churn (probability + time-to-event)
# ---------------------------

def generate_churn(df: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Risk score → probability; Weibull time → churn date, censored at cfg.end."""
    # Short names
    urban = df["urban"].values
    uses_app = df["uses_equity_mobile_app"].values
    uses_equitel = df["uses_equitel"].values
    mpesa_linked = df["mpesa_linked_to_bank"].values
    sessions = df["equity_mobile_sessions_last_30d"].values
    mob_txn = df["equity_mobile_txn_count_last_90d"].values
    vol_90d = df["equity_mobile_trans_volume_last_90d"].values
    bal_6m = df["avg_balance_last_6m"].values
    products = (
        # quick-and-dirty proxy: more activity & balances → likely holding more products
        1 + (mob_txn > 8).astype(int) + (bal_6m > 50_000).astype(int) + (vol_90d > 60_000).astype(int)
    )
    complaints = df["complaints_count_last_year"].values
    branch = df["branch_visits_count_last_year"].values
    repay = df["loan_repayment_rate"].values

    # Linear risk score (tunable weights; negative lowers churn risk)
    score  = 0.40 * (1 - urban)
    score += 0.30 * (complaints.clip(0, 3) > 0)
    score += 0.05 * (branch > 3)
    score -= 0.40 * uses_app
    score -= 0.20 * uses_equitel
    score -= 0.25 * mpesa_linked
    score -= 0.002 * sessions
    score -= 0.015 * mob_txn
    score -= 0.000010 * vol_90d
    score -= 0.000020 * bal_6m
    score -= 0.05 * products
    score -= 0.30 * repay
    score += rng.normal(0, 0.20, size=df.shape[0])  # unobserved noise

    p = sigmoid(score)

    # Hit ~30% churn: threshold at the 70th percentile of risk - hard to reach but not impossible
    cut = np.quantile(p, 1 - cfg.churn_target)
    churn_flag = (p >= cut).astype(int)

    # Time-to-churn via Weibull; higher risk → shorter scale (i.e., faster churn)
    scale_days = cfg.weibull_scale_days / (0.5 + p)
    t_days = (rng.weibull(cfg.weibull_k, size=df.shape[0]) * scale_days).astype(int)

    # Churn date = account open + T; censor at END
    end_dt = as_dt(cfg.end)
    acct_open = df["account_open_date"].values.astype("datetime64[D]")
    churn_date = acct_open + t_days.astype("timedelta64[D]")

    # If churn would happen after the window, treat as non-churn
    churned = churn_flag.copy()
    late_mask = (churned == 1) & (churn_date >= np.datetime64(end_dt.date()))
    churned[late_mask] = 0
    churn_date[~(churned == 1)] = np.datetime64("NaT")

    df = df.copy()
    df["churn_probability"] = p
    df["churned"] = churned
    df["churn_date"] = churn_date
    return df

# ---------------------------
# Orchestrator
# ---------------------------

def assemble_dataset(cfg: Config) -> pd.DataFrame:
    """Run all stages and return the final DataFrame."""
    rng = rng_from_seed(cfg.seed)

    df = generate_population(cfg, rng)
    df = generate_adoption(df, cfg, rng)
    df = generate_counts(df, cfg, rng)
    df = generate_amounts(df, cfg, rng)
    df = generate_credit(df, cfg, rng)
    df = generate_churn(df, cfg, rng)

    # Nice ordering (purely cosmetic)
    cols_order = [
        "customer_id", "account_open_date",
        "age", "gender", "region", "urban",
        "education_level", "employment_status", "KYC_verified",
        "uses_equity_mobile_app", "uses_equitel", "mpesa_linked_to_bank",
        "equity_mobile_sessions_last_30d", "equity_mobile_txn_count_last_90d",
        "mpesa_cash_in_count", "mpesa_cash_out_count",
        "num_deposits_last_90d", "num_withdrawals_last_90d",
        "equity_mobile_trans_volume_last_90d", "avg_balance_last_6m",
        "mshwari_savings_balance", "mshwari_loans_count", "fuliza_overdraft_amt",
        "loan_applications_count", "loan_repayment_rate",
        "branch_visits_count_last_year", "complaints_count_last_year",
        "churn_probability", "churned", "churn_date",
    ]
    # Use intersection so we don't crash if you add/remove columns
    cols_order = [c for c in cols_order if c in df.columns]
    return df[cols_order]