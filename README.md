# CarPricesPredition
A small Streamlit app and Jupyter notebook for predicting used car prices using multiple linear regression.

Project contents
- `car_price_app.py` — Streamlit application that trains a global linear regression model and optional per-brand models. It also provides an interactive UI to predict a car price from user inputs.
- `CarPrediction.ipynb` — Jupyter notebook for exploratory data analysis and experiments.
- `Cars_Dataset.csv` — Dataset used by the app and notebook.

Quick start (macOS / zsh)

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run car_price_app.py
```

Notes
- The app expects `Cars_Dataset.csv` to be in the same folder as `car_price_app.py`.
- By default the app trains a global Linear Regression model and will train brand-specific models if there are at least 20 samples for a given brand.
- The app displays dataset previews, correlation heatmaps, evaluation metrics (R², MAE, RMSE), coefficient importances, and an interactive prediction form.

Dependencies
- See `requirements.txt` for the packages used by the app and notebook (pandas, numpy, scikit-learn, streamlit, matplotlib, seaborn, etc.).

Development / Troubleshooting
- If you see encoding errors when reading the CSV, ensure the file is UTF-8 encoded or adjust `pd.read_csv()` in `car_price_app.py`.
- To re-run model training after editing the code, restart the Streamlit app or press "R" in the app to rerun.

License
- This repository does not include an explicit license file. Add one if you plan to publish this project.

If you'd like, I can also add a `requirements.txt` file and perform a quick syntax check on `car_price_app.py`.