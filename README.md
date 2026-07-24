# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions, built from scratch as a learning exercise — covering data exploration, preprocessing, model training, and evaluation.

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, containing 284,807 anonymized European credit card transactions, of which only 492 (0.17%) are fraudulent.

- `Time`, `Amount` — raw transaction features
- `V1`–`V28` — anonymized features (PCA-transformed for privacy)
- `Class` — target label (0 = genuine, 1 = fraud)

> Note: `creditcard.csv` is not included in this repo due to its size. Download it from the Kaggle link above and place it in the project root before running the notebook.

## Project Workflow

1. **Data Exploration** — inspected shape, class distribution, missing values, and the `Amount` distribution
2. **Preprocessing** — scaled `Amount` using `StandardScaler`, dropped the uninformative `Time` column
3. **Train/Test Split** — 80/20 split, stratified on the target to preserve the fraud ratio in both sets
4. **Model Training** — trained and compared two classifiers:
   - Logistic Regression (`class_weight='balanced'`)
   - Random Forest (100 trees, `class_weight='balanced'`)
5. **Evaluation** — assessed with precision, recall, and F1-score rather than raw accuracy, since accuracy is misleading on this heavily imbalanced dataset
6. **Inference** — saved the trained model and scaler with `joblib`, and built a script to predict on a single new transaction

## Results

| Metric (fraud class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.06 | **0.92** |
| Recall | **0.92** | 0.82 |
| F1-score | 0.11 | **0.86** |
| False positives | 1,441 | **7** |
| False negatives (missed fraud) | **8** | 18 |

**Random Forest was chosen as the final model** — it gives far fewer false alarms (7 vs 1,441) at a modest cost in recall, which is generally the better trade-off for a real-world fraud system where flooding customers with false fraud alerts erodes trust.

## Why not just use accuracy?

With fraud at only 0.17% of transactions, a model that predicts "genuine" for everything would still score ~99.8% accuracy while catching zero fraud. This project intentionally evaluates on precision/recall/F1 instead, since those actually measure whether fraud is being caught without excessive false alarms.

## Project Structure

```
credit-card-fraud-detection/
├── creditcard.csv                  # dataset (not included, download separately)
├── explore_data.ipynb              # data exploration, preprocessing, training, evaluation
├── predict_new_transaction.py      # loads saved model and predicts on a new transaction
├── fraud_detection_model.pkl       # saved trained Random Forest model
├── amount_scaler.pkl               # saved StandardScaler for the Amount feature
├── requirements.txt
└── README.md
```

## Setup & Usage

1. Clone the repo and download `creditcard.csv` from Kaggle into the project root
2. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run `explore_data.ipynb` cell by cell to reproduce the analysis and train the models
4. Run the inference script on a sample transaction:
   ```bash
   python predict_new_transaction.py
   ```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- joblib (model persistence)

## Learnings

This was built as a hands-on introduction to applied machine learning, with a focus on:
- Why accuracy alone fails on imbalanced classification problems
- The precision/recall trade-off and how to reason about it for a real use case
- The full ML pipeline: explore → preprocess → split → train → evaluate → serve

## License

This project is for educational purposes. The dataset is provided by [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle.
