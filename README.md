# 🍄 Mushroom Classification

A machine learning project that predicts whether a mushroom is **edible or poisonous** based on its physical characteristics. Built as part of a 6th semester Python/ML coursework using the UCI Mushroom dataset.

---

## What This Project Does

Given a set of physical features of a mushroom (like cap shape, color, odor, gill type, etc.), the model figures out whether it's safe to eat or not. It's a binary classification problem — edible (`e`) or poisonous (`p`).

We trained and compared four different classifiers on the same dataset to see which one performs best, and also built a simple interactive predictor where you can manually enter mushroom features and get a prediction back.

---

## Dataset

- **File:** `data/mushrooms.csv`
- **Source:** UCI Machine Learning Repository — Mushroom Dataset
- **Size:** ~8,000+ samples, 23 features (all categorical)
- **Target column:** `class` — either `e` (edible) or `p` (poisonous)

The features cover things like:
- Cap shape, surface, and color
- Odor
- Gill attachment, spacing, size, and color
- Stalk shape and surface
- Ring and veil details
- Spore print color
- Habitat and population

---

## Project Workflow

The notebook (`project.ipynb`) is structured step by step:

### 1. Importing Libraries
Standard data science stack — `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.

### 2. Data Loading
Loads `mushrooms.csv` into a DataFrame and takes a first look at the shape and sample rows.

### 3. Data Cleaning
- Checks for null values (there are none in this dataset)
- Removes duplicate rows
- Verifies class distribution

### 4. Exploratory Data Analysis (EDA)
- Class distribution plot (edible vs poisonous counts)
- Correlation heatmap across all encoded features

### 5. Preprocessing / Label Encoding
Since all features are categorical (stored as letters), we apply `LabelEncoder` to every column to convert them into numeric form. Each column gets its own encoder, which we save in a dictionary for later use in predictions.

### 6. Feature / Target Split
- **X** = all columns except `class`
- **y** = the `class` column (0 = edible, 1 = poisonous)

### 7. Train-Test Split
70% training / 30% testing, with `stratify=y` to maintain class balance in both splits.

### 8. Model Training & Evaluation

Four models were trained and evaluated:

| Model | Notes |
|---|---|
| **Logistic Regression** | Uses scaled data (`StandardScaler`), max_iter=1000 |
| **Random Forest** | 100 estimators, no scaling needed |
| **Decision Tree** | Default depth, no scaling needed |
| **SVM** | Linear kernel, uses scaled data |

Each model is evaluated with:
- Accuracy score
- Full classification report (precision, recall, F1)
- Confusion matrix (plotted as a heatmap)

### 9. Model Comparison
A bar chart comparing accuracy across all four models side by side. Also includes a grouped bar chart comparing Accuracy, Recall, and F1-score together.

### 10. Interactive Predictor
A `predict_mushroom()` function that walks you through entering each feature one by one (with hints showing valid options), then runs it through the trained model and tells you if the mushroom is edible or poisonous.

---

## Results

All four models perform extremely well on this dataset (the mushroom dataset is relatively clean and well-separated). Here's a rough summary:

| Model | Accuracy |
|---|---|
| Random Forest | ~100% |
| Decision Tree | ~100% |
| Logistic Regression | ~95%+ |
| SVM (linear) | ~98%+ |

> Random Forest and Decision Tree tend to get near-perfect accuracy because odor alone is a very strong predictor in this dataset.

---

## How to Run

**1. Clone or download the repo**

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Open the notebook**
```bash
jupyter notebook project.ipynb
```

**4. Run all cells** — they're numbered and meant to be run top to bottom.

**5. Try the predictor** — at the end of the notebook, call:
```python
predict_mushroom()
```
and enter your mushroom's features when prompted.

---

## Project Structure

```
python/
├── data/
│   ├── mushrooms.csv        # Main dataset used in this project
│   ├── diabetes-1.csv       # Other datasets (not used here)
│   └── lr-Real-estate(1).csv
├── project.ipynb            # Main Jupyter notebook
└── virtual_env/             # Python virtual environment
```

---

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Notes

- The dataset has no missing values, so no imputation was needed.
- All features are categorical — label encoding was the preprocessing step here.
- Scaled data (via `StandardScaler`) was only used for Logistic Regression and SVM, since tree-based models don't need it.
- The `encoders` dictionary is preserved after preprocessing so the interactive predictor can encode new inputs the same way the training data was encoded.
