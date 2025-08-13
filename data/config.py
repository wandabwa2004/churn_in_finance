# config.py
# All the knobs live here so you can tune the simulation
# without digging through the generator code.

from dataclasses import dataclass

@dataclass
class Config:
    # Core dataset size and window
    n: int = 10_000
    start: str = "2022-01-01"     # observation window start - you can adjust this to simulate different periods
    end: str   = "2025-01-01"     # observation window end - adjust to control how long the dataset spans
    seed: int  = 42               # reproducible runs
    churn_target: float = 0.50    # aim for ~30% churn observed within window

    # Region mix (rough proportions; feel free to tweak)
    region_weights: dict = None
    # Urban share by region (Nairobi ~ fully urban; others are mixed)
    urban_share: dict = None

    # Overdispersion parameter for Negative Binomial sampling
    nb_r: float = 2.0

    # Monetary distribution settings (medians + skew)
    median_vol_urban: float = 60_000.0
    median_vol_rural: float = 40_000.0
    sigma_vol: float = 0.60

    median_bal_urban: float = 45_000.0
    median_bal_rural: float = 28_000.0
    sigma_bal: float = 0.70

    # Fuliza / M-Shwari
    fuliza_mean_base: float = 2_800.0
    mshwari_mean_base: float = 5_000.0
    mshwari_zero_prob: float = 0.50

    # Weibull churn timing
    weibull_k: float = 1.4                 # k>1 → increasing hazard over time
    weibull_scale_days: float = 365 * 1.2  # “typical” time-to-churn scale - shortened to make sure some  sizeable churn is observed within window. Play with this to adjust churn timing

    def __post_init__(self):
        if self.region_weights is None:
            self.region_weights = {
                "Nairobi": 0.22,
                "Rift Valley": 0.24,
                "Coast": 0.14,
                "Western": 0.16,
                "Central": 0.16,
                "Northeastern": 0.08,
            }
        if self.urban_share is None:
            self.urban_share = {
                "Nairobi": 1.00,
                "Rift Valley": 0.45,
                "Coast": 0.55,
                "Western": 0.40,
                "Central": 0.60,
                "Northeastern": 0.35,
            }
