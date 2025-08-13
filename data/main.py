# main.py - Tiny runner. Generates the dataset and writes it to CSV.

from config import Config
from generator import assemble_dataset

if __name__ == "__main__":
    cfg = Config(
        n=10_000,
        start="2022-01-01",
        end="2025-01-01",
        seed=42,
        churn_target=0.50,  # aim for ~30% observed churn
    )

    df = assemble_dataset(cfg)

    # Quick sanity print
    observed_churn = df["churned"].mean()
    print(f"Rows: {len(df):,} | Observed churn: {observed_churn:.3f}")

    out_path = "equity_bank_churn_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")