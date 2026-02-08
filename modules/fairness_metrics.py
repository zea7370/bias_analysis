from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)

def compute_fairness(y_true, y_pred, sensitive_features):
    metrics = {}

    metrics["Demographic Parity Difference"] = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    metrics["Disparate Impact"] = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    metrics["Equal Opportunity Difference"] = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    return metrics
