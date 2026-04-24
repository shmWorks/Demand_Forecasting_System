# Retail-IQ: Professional Retail Forecasting System

## 📌 Overview

Retail-IQ is a machine learning-based sales forecasting system designed to handle large-scale retail time-series data. It features a modular architecture for data preprocessing, feature engineering, and automated reporting, providing real-time predictions through a user-friendly interface.

## 👥 Team Members

- Ayesha Khalid (23L-0667)
- Uma E Rubab (23L-0928)
- Sheraz Malik (23L-0572)

## 🛠 Tech Stack

- **Machine Learning**: Scikit-learn, XGBoost, Statsmodels
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Backend**: Python (Flask)
- **Frontend**: HTML, CSS, Bootstrap
- **Database**: SQL Server / MySQL
- **DevOps/Tools**: Pathlib (robust paths), uv/Setuptools (package mgmt)

## 📂 Project Structure

```
Retail-IQ/
├── data/
│   ├── raw/                # Unmodified input CSVs
│   └── processed/          # Cleaned and featured datasets
├── docs/                   # Project documentation and reports
├── notebooks/              # Jupyter notebooks for experimentation
├── outputs/
│   ├── figures/            # Generated plots and visualizations
│   ├── models/             # Serialized model files (.pkl, .json)
│   └── logs/               # Processing and error logs
├── src/                    # Core Python package (Modular logic)
│   └── retail_iq/
│       ├── config.py       # Centralized path and config management
│       ├── preprocessing.py # Data loading and cleaning
│       ├── features.py      # Feature engineering logic
│       └── visualization.py # Plotting utilities
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project metadata and dependencies
└── requirements.txt        # Dependency list
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .

# Using pip
pip install -r requirements.txt
pip install -e .
```

### 2. Usage

#### Data Analysis & EDA

Open `notebooks/eda.ipynb` to run the exploration, feature engineering, and baseline model training.

#### Web Application

The Flask-based interface can be launched once the core package is installed. (Check `app/` or `run.py` in previous versions for entry points).

---
