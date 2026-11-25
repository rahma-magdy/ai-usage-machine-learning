# AI Usage: Machine Learning Mini-Project

## Project Overview

This mini-project explores an AI Usage dataset and applies basic data
preparation, feature engineering, and two introductory predictive models. It
uses the `Productivity_Score` column in two different ways:

1. **Regression:** predict the numerical productivity score from daily AI
   usage hours.
2. **Classification:** classify a record as **Low** or **High Productivity**
   using a teaching threshold of 6. Scores below 6 are Low; scores of 6 or
   higher are High.

## Project files

| File | Purpose |
|---|---|
| `clean_ai_usage_data.py` | Cleans `AI_Usage.csv`, handles missing and duplicate values, and writes `cleaned_AI_Usage.csv`. |
| `explore_ai_usage_data.py` | Reads the cleaned data and displays charts and correlations. |
| `Feature_Engineering.py` | Adds age groups, standardizes selected category values, one-hot encodes categorical columns, and writes feature-engineered and encoded datasets. |
| `ai_usage_linear_regression.py` | Uses linear regression to predict the numerical `Productivity_Score` from daily AI usage hours and reports mean absolute error. |
| `ai_usage_neural_network.py` | Uses a neural network and age plus daily AI usage hours to classify productivity: scores below 6 are Low (0), and scores of 6 or higher are High (1). |

## Data files

- `AI_Usage.csv` is the input used by the cleaning script.
- `cleaned_AI_Usage.csv` is produced by `clean_ai_usage_data.py` and used by the analysis and modeling scripts.
- `AI_Usage_feature_engineered.csv` and `AI_Usage_encoded.csv` are produced by `Feature_Engineering.py`.

The scripts expect these CSV files to be in the current working directory.

## Requirements

Python 3 and the following packages are used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install the packages with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the project

Run commands from the project directory:

```bash
python clean_ai_usage_data.py
```

This creates `cleaned_AI_Usage.csv`. Then run any of the following scripts:

```bash
python explore_ai_usage_data.py
python Feature_Engineering.py
python ai_usage_linear_regression.py
python ai_usage_neural_network.py
```

The visualization script opens plots. The modeling scripts print evaluation
results in the terminal. Run `Feature_Engineering.py` after
`clean_ai_usage_data.py`, since it uses the cleaned dataset.
