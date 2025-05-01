# Used-Vehicle-Selling-Price-Prediction-Applied-Machine-Learning

This machine learning project predicts used vehicle selling prices using various attributes such as make, model, year, transmission, and mileage. Leveraging both regression and ensemble learning techniques, the goal is to enable accurate pricing forecasts that can assist automotive stakeholders in production, inventory, and marketing decisions.

## Project Overview

Accurate used car price prediction is a valuable task for manufacturers, dealers, and consumers. This project uses a publicly available Kaggle dataset with over 500,000 listings and applies multiple modeling strategies—ranging from baseline regression to advanced ensemble methods like Random Forests.

## Dataset

- **Source**: [Vehicle Sales Dataset](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data?resource=download)
- **Initial Size**: 558,837 entries, 16 attributes
- **After Cleaning**: 533,648 entries, 13 attributes
- **Target Variable**: `sellingprice`

## Technologies Used

- Python, Jupyter Notebook
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (Linear Regression, Lasso, Polynomial Regression, Random Forest)
- Feature Engineering: One-Hot Encoding, Frequency Encoding, Ordinal Encoding
- Scaling Techniques: MinMaxScaler, StandardScaler

## Key Features

- Cleaned and normalized raw vehicle listings
- Performed data encoding and transformation for model readiness
- Conducted model comparisons: linear regression, regularization, and random forest
- Explored feature interactions such as combined "model + trim" attributes

## Modeling Approaches and Results

### Linear Regression Models
| Model | Approach | R² Score | Notes |
|-------|----------|----------|-------|
| Model 1 | Numerical-only Linear Regression | 0.39 | Baseline model |
| Model 2 | Polynomial Regression | 0.43 | Higher complexity |
| Model 3 | One-Hot Encoding | -3.44 | Severe overfitting |
| Model 4 | Lasso Regularized (One-Hot) | 0.699 | 30% improvement |
| Model 5 | Frequency Encoding | 0.45 | Only 12 features used |

**Conclusion:** Model 4 (Lasso with One-Hot Encoding) performed best among linear models but required careful feature pruning. Model 5 offered a tradeoff between interpretability and performance.

### Random Forest Models
| Approach | R² Score | Notes |
|----------|----------|-------|
| Ordinal Encoding + Separate "model" & "trim" | 0.8161 | Good baseline |
| Ordinal Encoding + Combined "model_trim" | 0.9543 | Best model overall |

**Insights:**
- Feature engineering and categorical encoding significantly improved performance
- Combining key categorical variables helped the model learn richer interactions

## Business Impact

Accurate pricing models enable:
- **Optimized inventory and production planning**
- **Targeted marketing and budget allocation**
- **Improved supply chain efficiency and cost savings**

## How to Run

1. Clone the repository and open the notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
