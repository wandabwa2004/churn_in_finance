"""
Simple churn prediction pipeline with two models (Logistic Regression + Gradient Boosting).

Run:
    python churn_training_simple.py
"""

import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)

import seaborn as sns

# -----------------------------
# 1) Settings
# -----------------------------
DATA_PATH = "../data/equity_bank_churn_dataset.csv"
AS_OF_CUTOFF = "2024-01-01"
TARGET = "churned"
SEED = 42
MODEL_DIR = "models"

DROP_COLS = [
    "customer_id",
    "churn_probability",  # synthetic leakage
    "churn_date",         # leakage
]
DATE_COLS = ["account_open_date"]

# -----------------------------
# 2) Helpers
# -----------------------------
def parse_dates(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    return out

def temporal_split(df, cutoff_str):
    cutoff = pd.to_datetime(cutoff_str)
    train = df[df["account_open_date"] < cutoff].copy()
    test  = df[df["account_open_date"] >= cutoff].copy()
    return train, test

def describe_dataset(X, y, dates, name):
    print(f"\n=== {name.upper()} DATA ===")
    print(f"Rows: {len(X):,} | Columns: {X.shape[1]}")
    print(f"Date range: {dates.min().date()} → {dates.max().date()}")
    print(f"Churn rate: {y.mean()*100:.2f}%")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        print("\nNumeric feature ranges:")
        print(X[num_cols].agg(['min', 'max', 'mean']).T)

    if cat_cols:
        print("\nMost frequent categories:")
        for c in cat_cols:
            top_cat = X[c].value_counts(normalize=True).head(1)
            top_val, top_pct = top_cat.index[0], top_cat.iloc[0]*100
            print(f"  {c}: {top_val} ({top_pct:.1f}%)")

def top_k_precision(y_true, y_score, k=0.10):
    n = len(y_true)
    k_n = max(1, int(k * n))
    idx = np.argsort(-y_score)[:k_n]
    return float(y_true[idx].mean())

def evaluate_model(name, model, X_test, y_test):
    """Return dict of metrics and confusion matrix for a fitted pipeline."""
    proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    top10 = top_k_precision(y_test, proba, 0.10)
    top20 = top_k_precision(y_test, proba, 0.20)

    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "top10_precision": top10,
        "top20_precision": top20,
        "conf_matrix": cm
    }

# -----------------------------
# 3) Load and split
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = parse_dates(df, DATE_COLS)

y = df[TARGET].astype(int).values
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET], errors="ignore")

# Only add columns that are not already in X
cols_to_add = [col for col in DATE_COLS + [TARGET] if col not in X.columns]
df_for_split = pd.concat([X, df[cols_to_add]], axis=1)
train_df, test_df = temporal_split(df_for_split, AS_OF_CUTOFF)

X_train = train_df.drop(columns=DATE_COLS + [TARGET])
y_train = train_df[TARGET].astype(int).values
X_test  = test_df.drop(columns=DATE_COLS + [TARGET])
y_test  = test_df[TARGET].astype(int).values

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# Describe training and test sets
describe_dataset(X_train, y_train, train_df["account_open_date"], "Train")
describe_dataset(X_test, y_test, test_df["account_open_date"], "Test")

# -----------------------------
# 4) Preprocessing
# -----------------------------
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

pre = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols),
])

# -----------------------------
# 5) Models
# -----------------------------
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
gbdt   = GradientBoostingClassifier(
    random_state=SEED,
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3,
    subsample=0.9
)

pipe_lr  = Pipeline([("pre", pre), ("clf", logreg)])
pipe_gbdt = Pipeline([("pre", pre), ("clf", gbdt)])

# -----------------------------
# 6) Train
# -----------------------------
pipe_lr.fit(X_train, y_train)
pipe_gbdt.fit(X_train, y_train)

# -----------------------------
# 7) Evaluate
# -----------------------------
results = []
results.append(evaluate_model("Logistic Regression", pipe_lr, X_test, y_test))
results.append(evaluate_model("Gradient Boosting", pipe_gbdt, X_test, y_test))

print("\n=== MODEL COMPARISON (Test Set) ===")
for r in results:
    print(f"{r['name']}:")
    print(f"  ROC AUC:        {r['roc_auc']:.4f}")
    print(f"  PR  AUC:        {r['pr_auc']:.4f}")
    print(f"  Top 10% Prec.:  {r['top10_precision']:.4f}")
    print(f"  Top 20% Prec.:  {r['top20_precision']:.4f}")
    print(f"  Confusion Matrix @0.5 [TN FP; FN TP]:\n{r['conf_matrix']}")
    print("-" * 50)




def plot_gains_and_lift(y_true, y_score, model_name):
    """
    Plots cumulative gains and lift chart for a given model.
    """
    # Sort by score descending
    order = np.argsort(-y_score)
    y_true_sorted = np.array(y_true)[order]
    
    # Cumulative churners captured
    cum_churn = np.cumsum(y_true_sorted)
    total_churn = y_true.sum()
    cum_perc_churn = cum_churn / total_churn
    
    # Population percentages
    n = len(y_true)
    perc_pop = np.arange(1, n + 1) / n
    
    # Lift = (percentage of churners captured) / (percentage of population targeted)
    lift = cum_perc_churn / perc_pop

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Gains Chart ---
    axes[0].plot(perc_pop * 100, cum_perc_churn * 100, label=model_name)
    axes[0].plot([0, 100], [0, 100], 'k--', label='Random')
    axes[0].set_title(f'Cumulative Gains - {model_name}')
    axes[0].set_xlabel('Population targeted (%)')
    axes[0].set_ylabel('Churners captured (%)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Lift Chart ---
    axes[1].plot(perc_pop * 100, lift, label=model_name)
    axes[1].plot([0, 100], [1, 1], 'k--', label='Random')
    axes[1].set_title(f'Lift Chart - {model_name}')
    axes[1].set_xlabel('Population targeted (%)')
    axes[1].set_ylabel('Lift')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()



def get_feature_names(pipeline, X_train):
    preprocessor = pipeline.named_steps['pre']

    # Get feature names for numeric columns
    num_features = []
    cat_features = []
    cat_encoded = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            num_features = cols
        elif name == "cat":
            # If the transformer is a pipeline, get the OneHotEncoder
            if hasattr(transformer, 'named_steps') and 'ohe' in transformer.named_steps:
                cat_features = cols
                cat_encoder = transformer.named_steps['ohe']
                cat_encoded = cat_encoder.get_feature_names_out(cat_features)
    
    return np.concatenate([num_features, cat_encoded])


# --- Logistic Regression Feature Importance ---
def plot_lr_coefficients(pipe, X_train, top_n=15):
    feature_names = get_feature_names(pipe, X_train)
    coefs = pipe.named_steps['clf'].coef_[0]  # <-- fix here

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_importance": np.abs(coefs)
    }).sort_values(by="abs_importance", ascending=False).head(top_n)

    print("\n=== Top Features - Logistic Regression ===")
    print(importance_df[['feature', 'coefficient']])

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="coefficient", 
        y="feature", 
        data=importance_df,
        palette=["#d62728" if x > 0 else "#1f77b4" for x in importance_df["coefficient"]]
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Top {top_n} Features - Logistic Regression\n(+ increases churn, – reduces churn)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return importance_df


# --- Gradient Boosting Feature Importance ---
def plot_gbdt_importance(pipe, X_train, top_n=15):
    feature_names = get_feature_names(pipe, X_train)
    importances = pipe.named_steps['clf'].feature_importances_  # <-- fix here

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    print("\n=== Top Features - Gradient Boosting ===")
    print(importance_df)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="importance", 
        y="feature", 
        data=importance_df,
        color="#2ca02c"
    )
    plt.title(f"Top {top_n} Features - Gradient Boosting")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return importance_df

proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
proba_gbdt = pipe_gbdt.predict_proba(X_test)[:, 1]

plot_gains_and_lift(y_test, proba_lr, "Logistic Regression")
plot_gains_and_lift(y_test, proba_gbdt, "Gradient Boosting")




# --- Run for both models ---
lr_importance_df = plot_lr_coefficients(pipe_lr, X_train)
gbdt_importance_df = plot_gbdt_importance(pipe_gbdt, X_train)


# -----------------------------
# 8) Pick the best model
# -----------------------------
# Here we pick by ROC AUC; change key to "pr_auc" if you prefer
best_model = max(results, key=lambda x: x["roc_auc"])
print(f"\nBest model by ROC AUC → {best_model['name']}")

# -----------------------------
# 9) Save the best pipeline
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
if best_model["name"] == "Logistic Regression":
    joblib.dump(pipe_lr, os.path.join(MODEL_DIR, "best_model.joblib"))
else:
    joblib.dump(pipe_gbdt, os.path.join(MODEL_DIR, "best_model.joblib"))

print(f"Saved best model pipeline → {MODEL_DIR}/best_model.joblib")