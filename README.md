# mlops-assignment

A machine learning project built with numpy, pandas, and scikit-learn, managed using a conda environment.

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

## Running the Code

After activating the environment, run any Python script with:

```bash
python <script_name>.py
```

For example:

```bash
python main.py
```

---

## Dependencies

All required packages are listed in [`requirements.txt`](requirements.txt). Key libraries:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `scikit-learn` | Machine learning |
