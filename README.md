# Movie Genre Classification — CodTech Internship Project. Intership ID-CITS3857

Classifies movies into 8 genres using **TF-IDF + Logistic Regression** on movie
titles and plot overviews. No external dataset download required.

## Genres Supported
Action · Comedy · Drama · Horror · Romance · Sci-Fi · Thriller · Animation

## Project Structure

```
movie_genre/
├── generate_data.py      ← creates the 96-movie dataset
├── preprocess.py         ← cleans text, runs EDA, generates plots
├── train_model.py        ← trains TF-IDF + Logistic Regression + evaluation
├── predict.py            ← classify new movies (demo + interactive mode)
├── run_pipeline.py       ← runs all steps in one command
├── requirements.txt
├── data/                 ← generated CSVs
├── models/               ← saved model, encoder, metrics
└── outputs/              ← PNG charts
```

## Setup & Run

```bash
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py

# Or step by step
python generate_data.py
python preprocess.py
python train_model.py
python predict.py

# Interactive mode — classify your own movie descriptions
python predict.py --interactive
```

## How It Works

1. **Data** — 96 manually curated movies (12 per genre) with titles and plot overviews.

2. **Preprocessing** — lowercasing, punctuation removal, stopword filtering applied to
   `title + overview` combined text.

3. **Feature extraction** — TF-IDF with unigrams + bigrams, 5000 features, sublinear
   TF scaling.

4. **Model** — Multinomial Logistic Regression (lbfgs solver, C=1.0).

5. **Evaluation** — 5-fold stratified cross-validation, confusion matrix, per-class F1.

## Outputs

| File | Description |
|---|---|
| `outputs/1_eda.png` | Genre distribution + overview length histogram |
| `outputs/2_top_words.png` | Top 15 TF-IDF keywords per genre |
| `outputs/3_confusion_matrix.png` | Confusion matrix on test set |
| `outputs/4_cv_scores.png` | 5-fold CV accuracy per fold |
| `outputs/5_per_class_f1.png` | F1-score bar chart per genre |
| `models/metrics.json` | All numeric metrics |

## Extending the Project

- **More data**: drop in a Kaggle TMDB CSV and point `generate_data.py` to it
- **Better model**: swap `LogisticRegression` for `SVC` or `RandomForest` in `train_model.py`
- **Deep learning**: replace TF-IDF with a sentence-transformers embedding
