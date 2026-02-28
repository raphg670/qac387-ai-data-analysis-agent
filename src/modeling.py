from __future__ import annotations

from typing import Optional, List, Dict, Any
import pandas as pd
import statsmodels.api as sm


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Fit a multiple linear regression model and return JSON-safe results."""
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found.")

    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError("Outcome must be numeric.")

    if predictors is None:
        predictors = df.select_dtypes(include=["number"]).columns.tolist()
        predictors = [c for c in predictors if c != outcome]

    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor '{p}' not found in dataframe.")

    if len(predictors) == 0:
        raise ValueError("No predictors available for regression.")

    cols_needed = [outcome] + predictors
    clean_df = df[cols_needed].dropna()

    X = clean_df[predictors]
    y = clean_df[outcome]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    intercept = float(model.params["const"])
    coefficients = {str(k): float(v) for k, v in model.params.drop("const").items()}

    results = {
        "outcome": str(outcome),
        "predictors": [str(p) for p in predictors],
        "n_rows_used": int(clean_df.shape[0]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": intercept,
        "coefficients": coefficients,
    }
    return results