# mlops-assignment

A machine learning project using the Boston Housing dataset, built with numpy, pandas, and scikit-learn, managed using a conda environment.

---

## Repository

**GitHub:** https://github.com/khateeb-haq/mlops-assignment

---

## Branches

| Branch | Description |
|---|---|
| `main` | Main branch — merged from all feature branches |
| `dtree` | DecisionTreeRegressor model (`train.py`) |
| `kernelridge` | KernelRidge model (`train2.py`) + GitHub Actions workflow |

---

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.12

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/khateeb-haq/mlops-assignment.git
cd mlops-assignment
```

### 2. Create the conda environment

```bash
conda create -n mlops-assignment python=3.12 -y
conda activate mlops-assignment
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
mlops-assignment/
├── .github/
│   └── workflows/
│       └── train.yml     # GitHub Actions workflow (triggered on push to kernelridge)
├── train.py              # DecisionTreeRegressor training script
├── train2.py             # KernelRidge training script
├── misc.py               # Reusable helper functions (load, preprocess, train, evaluate)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Running the Code

### Activate the conda environment

```bash
conda activate mlops-assignment
```

### DecisionTreeRegressor (`dtree` branch)

```bash
git checkout dtree
python train.py
```

**Expected output:**
```
Dataset loaded: 506 rows, 14 columns
Train size: 404 | Test size: 102
Model training complete.

Model     : DecisionTreeRegressor
MSE (test): 10.4161
Avg CV MSE: 38.3089
```

### KernelRidge (`kernelridge` branch)

```bash
git checkout kernelridge
python train2.py
```

**Expected output:**
```
Dataset loaded: 506 rows, 14 columns
Train size: 404 | Test size: 102
Model training complete.

Model     : KernelRidge
MSE (test): 476.2581
Avg CV MSE: 584.6861
```

---

## Performance Comparison

| Model | MSE (Test Set) | Avg CV MSE (5-fold) |
|---|---|---|
| `DecisionTreeRegressor` | 10.4161 | 38.3089 |
| `KernelRidge` (RBF kernel) | 476.2581 | 584.6861 |

---

## GitHub Actions

A workflow is configured at `.github/workflows/train.yml` that triggers on every push to the `kernelridge` branch. It:

1. Checks out the code
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs `train.py` (DecisionTreeRegressor)
5. Runs `train2.py` (KernelRidge)

---

## How It Works

- **`misc.py`** contains generic, reusable functions for any sklearn regression model:
  - `load_data()` — fetches the Boston Housing dataset from http://lib.stat.cmu.edu/datasets/boston
  - `preprocess_data()` — splits data into train/test sets
  - `train_model()` — fits any sklearn-compatible model
  - `evaluate_model()` — computes MSE on test data
  - `average_cv_mse()` — computes average MSE via 5-fold cross-validation

- **`train.py`** uses those functions to train a `DecisionTreeRegressor` and reports MSE.
- **`train2.py`** uses those same functions to train a `KernelRidge` model and reports MSE.

---

## Dependencies

All required packages are listed in [`requirements.txt`](requirements.txt). Key libraries:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning |
