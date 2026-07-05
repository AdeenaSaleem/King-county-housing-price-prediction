# King County Housing Price Prediction — Investment Screening Model

A regression analysis and modeling project built for a (hypothetical) Real Estate Investment
Trust (REIT) evaluating residential property purchases in the Seattle metro area.

## Business Problem

A REIT wants a fast, data-driven way to estimate whether a listed home price is reasonable given
its features, before committing to further due diligence. This project analyzes ~21,600 King
County home sales (May 2014 – May 2015) to identify the strongest price drivers, and builds and
compares several predictive models to determine which one is best suited as a first-pass
screening tool.

## Key Questions

1. Which property features most influence price, and are any associated with unusually high
   valuation risk (large price swings)?
2. How much does prediction accuracy improve as models move from a single feature to multiple
   features, engineered/polynomial features, regularization, and a non-linear ensemble model?
3. What model is reasonable to hand to an analyst as a screening tool, and what are its limits?

## Approach

- **Data cleaning:** dropped non-predictive identifier columns; imputed missing `bedrooms` /
  `bathrooms` values with the column mean.
- **EDA:** correlation analysis, floor-count distribution, waterfront price/outlier comparison
  (boxplot), and `sqft_above` vs. price relationship (regression plot).
- **Modeling, in order of complexity:**
  - Single-feature linear regression (`long`, then `sqft_living`) as baselines
  - Multiple linear regression across 11 features
  - Scaling + polynomial feature engineering pipeline
  - Ridge regression (L2-regularized) on plain and polynomial features, evaluated on a held-out
    test set
  - Random Forest as a non-linear benchmark, with feature importance analysis
- **Evaluation:** R² and MSE on unseen test data (not training data) throughout, to assess real
  generalization rather than training fit alone.

## Key Findings

- `sqft_living`, `grade`, and location (`lat`) are consistently the strongest price drivers
  across every model tested — from simple correlation through Random Forest feature importance.
- A single feature (`sqft_living`) explains roughly **49%** of price variance; combining features
  and adding non-linear structure pushes test-set R² up to **~0.70–0.78** depending on the model.
- The **Random Forest model outperformed all linear/regularized models on the held-out test set**,
  suggesting real non-linear structure in how size, quality, and location interact to drive price.
- Waterfront homes show a higher median price and wider typical range; non-waterfront homes show
  more extreme individual outliers relative to their own distribution — a risk signal worth
  flagging separately rather than smoothing into a single point estimate.

## Recommendation

Use the Random Forest model (or the simpler, more interpretable Ridge + polynomial model) as an
automated first-pass screening tool to flag whether a listed price looks reasonable — supporting,
not replacing, human due diligence. Predictions are notably less reliable at the high end of the
market, where unique/luxury features not captured in this dataset likely matter more.

## Tech Stack

Python · pandas · NumPy · scikit-learn · seaborn · matplotlib

## Repo Contents

- `King_County_Housing_Price_Prediction_Portfolio.ipynb` — full analysis notebook (data cleaning,
  EDA, modeling, evaluation, findings)
- `README.md` — this file

## How to Run

1. Clone the repo and open the notebook in Jupyter.
2. Run all cells — the notebook pulls the dataset directly from its public source URL, no local
   files needed.

## Limitations & Next Steps

- Data spans a single year (2014–2015), so market cycles and seasonality across years aren't
  captured.
- Location is represented only by raw coordinates/zipcode; neighborhood-level features (e.g.
  median price per zipcode, school district, walkability) would likely improve accuracy further.
- Hyperparameter tuning (Ridge `alpha`, Random Forest depth/estimators) was kept minimal; a grid
  search would be a natural next step before any production use.
