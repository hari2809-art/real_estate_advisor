# Real Estate Investment Advisor

## Project Structure
- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for EDA, preprocessing, modeling
- `models/` - Saved ML models and encoders
- `app/` - Streamlit application
- `mlflow_logs/` - MLflow experiment tracking logs

## Run Order
1. notebooks/01_preprocessing.ipynb
2. notebooks/02_eda.ipynb
3. notebooks/03_modeling.ipynb
4. streamlit run app/app.py

## Install dependencies
pip install -r requirements.txt
