# Equity Bank Churn Simulation & Prediction

This project generates a realistic synthetic dataset for customer churn analysis in the banking sector, specifically modeled for a hypothetical African bank called Equity Bank. It includes configurable simulation parameters, modular data generation, and utilities for downstream machine learning tasks—including a full churn prediction pipeline.

---

## Features

- **Configurable simulation:** Easily adjust dataset size, time window, churn rate, region mix, and more via `config.py`.
- **Modular data generation:** Each stage (demographics, adoption, transactions, monetary amounts, credit, churn) is handled by a separate function in `generator.py`.
- **Reproducible results:** Set a random seed for consistent outputs.
- **Churn modeling:** Simulates churn probability and time-to-event using realistic business logic.
- **Churn prediction pipeline:** `model.py` trains and evaluates two models (Logistic Regression and Gradient Boosting), provides feature importance, and saves the best model.

---

## Environment Setup

This project uses Conda for environment management.  
You can recreate the exact environment using the provided `environment.yml` file.

### 1. Clone the repository

```bash
git clone https://github.com/wandabwa2004/churn_in_finance.git
cd churn_in_finance/churn/equity/data
```

### 2. Create the environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate docs_39
```

### 4. Update the environment (if you add new packages)

```bash
conda env export > environment.yml
```

---

## Getting Started

### 1. Generate the synthetic dataset

Run the main script to create a synthetic dataset:

```bash
python main.py
```

This will output a CSV file named `equity_bank_churn_dataset.csv` in the current directory.

### 2. Train and evaluate churn prediction models

Move to the `py` directory and run the model training pipeline:

```bash
cd ../py
python model.py
```

This will:
- Train Logistic Regression and Gradient Boosting models
- Print evaluation metrics and feature importance
- Save the best model pipeline to the `models/` directory

---

## File Structure

```
data/
├── config.py        # Simulation configuration (parameters, weights, etc.)
├── generator.py     # Modular data generation functions
├── main.py          # Runner script to generate and save the dataset

py/
├── model.py         # Churn prediction pipeline (training, evaluation, feature importance)
├── models/          # Saved model pipelines
```

---

## Customization

- **Change simulation parameters:** Edit `config.py` or pass new values in `main.py`.
- **Add new features:** Extend `generator.py` with new functions or logic.
- **Tune models:** Modify model hyperparameters in `model.py`.

---

## Example Output

After running `main.py`, you’ll see a summary like:

```
Rows: 10,000 | Observed churn: 0.305
Saved: equity_bank_churn_dataset.csv
```

After running `model.py`, you’ll see model metrics and feature importance, and the best model will be saved as `models/best_model.joblib`.

---

## License

MIT License

---

**Note:**  
The generated data is synthetic and intended for educational, prototyping, and demonstration purposes only.
