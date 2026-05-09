with open("README.md", "r") as f:
    content = f.read()

new_content = """
## 🚀 Full Stack Dashboard Setup Guide

Retail-IQ now ships with a brutally minimalistic, high-performance web dashboard (Flask + MongoDB + Alpine.js + Chart.js) to monitor the data science pipeline execution and visualize forecasting metrics in real-time.

### 1. Prerequisites
- Python 3.10+
- MongoDB 7.0+ (Running locally or via Docker)
- [Kaggle API Credentials](https://github.com/Kaggle/kaggle-api) (for downloading raw data)

### 2. Environment & Dependencies

Activate your virtual environment and install the required dependencies (including the new web stack):

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .
```

*Required packages include: `flask`, `flask-cors`, `pymongo`, `pydantic`, `PyJWT`, `bcrypt`.*

### 3. Database Setup (MongoDB)

Ensure MongoDB is running on your system. If you are using Linux:

```bash
sudo systemctl start mongod
```

To populate the database with mock metrics for the dashboard, run the seeding script:

```bash
PYTHONPATH=. python3 dashboard/seed_db.py
```
*This will create the necessary collections, targeted indexes, and an initial `admin` user.*

### 4. Downloading Raw Data

The pipeline requires the Corporación Favorita dataset to execute. Download it directly from Kaggle into the `data/raw/` directory:

```bash
mkdir -p data/raw
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
unzip data/raw/store-sales-time-series-forecasting.zip -d data/raw
rm data/raw/store-sales-time-series-forecasting.zip
```

### 5. Running the Application

Start the Flask backend server:

```bash
PYTHONPATH=. python3 dashboard/app.py
```

- Navigate to `http://127.0.0.1:5000` in your web browser.
- Log in using the default credentials (Username: `admin` / Password: `admin` if generated via `seed_db.py`).
- Use the **Pipeline Execution Engine** panel to trigger the data processing, feature engineering, and EDA pipeline. Real-time artifacts will appear directly on the dashboard once completed.
"""

# Let's insert this before the Project Structure section
parts = content.split("## Project Structure")
final_content = parts[0] + new_content + "\n## Project Structure" + parts[1]

with open("README.md", "w") as f:
    f.write(final_content)
