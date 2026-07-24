# MovieIQ — Predictive Analytics on Film Success

An end-to-end ML project + Streamlit dashboard predicting whether a movie will
be commercially successful (revenue > budget) from its budget, popularity,
runtime, rating, and genre.

## Project structure

```
MovieIQ/
├── MovieIQ.py              # Streamlit dashboard (Stage 5)
├── analysis.py             # Stages 1-4: data prep, EDA, stats tests, model training
├── movies.csv               # raw input data
├── movies_clean.csv         # cleaned data + engineered features (generated)
├── results.json              # every computed number, read by the app + ANSWERS.md (generated)
├── requirements.txt
├── ANSWERS.md               # written answers to every project question
├── assets/                  # saved EDA/model charts (generated)
│   ├── budget_vs_revenue.png
│   ├── genre_trends.png
│   ├── feature_vs_success_boxplots.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── model/                   # trained model + encoders (generated)
    ├── random_forest.joblib
    ├── genre_encoder.joblib
    ├── feature_cols_base.joblib
    ├── all_feature_columns.joblib
    └── genre_classes.joblib
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce the analysis (Stages 1-4)

This regenerates `movies_clean.csv`, everything in `assets/`, everything in
`model/`, and `results.json`:

```bash
python analysis.py
```

## Run the dashboard (Stage 5)

```bash
streamlit run MovieIQ.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Deploying to Streamlit Community Cloud

1. Push this whole folder to a public (or connected private) GitHub repo.
2. Make sure `movies_clean.csv`, `results.json`, and `model/*.joblib` are
   committed — the app reads them at startup rather than recomputing on
   every run.
3. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file path to `MovieIQ.py`.
4. Streamlit Cloud installs `requirements.txt` automatically — no other
   config is needed for this project (no secrets/env vars required).
5. Paste the live URL into `ANSWERS.md` once deployed.
