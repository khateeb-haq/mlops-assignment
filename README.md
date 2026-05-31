# mlops-assignment

A machine learning project using the Boston Housing dataset, built with numpy, pandas, and scikit-learn, managed using a conda environment.

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
├── boston.csv        # Boston Housing dataset
├── train.py          # Main training script
├── misc.py           # Reusable helper functions (load, preprocess, train, evaluate)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Running the Code

### Switch to the correct branch

```bash
git checkout dtree
```

### Activate the conda environment

```bash
conda activate mlops-assignment
```

### Run the training script

```bash
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

---

## How It Works

- **`misc.py`** contains generic, reusable functions for any sklearn regression model:
  - `load_data()` — loads the CSV dataset
  - `preprocess_data()` — splits data into train/test sets
  - `train_model()` — fits any sklearn-compatible model
  - `evaluate_model()` — computes MSE on test data
  - `average_cv_mse()` — computes average MSE via 5-fold cross-validation

- **`train.py`** uses those functions to train a `DecisionTreeRegressor` on the Boston Housing dataset and reports the MSE.

---

## Dependencies

All required packages are listed in [`requirements.txt`](requirements.txt). Key libraries:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning |
