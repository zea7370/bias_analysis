from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

def mitigate_bias(X_train, y_train, sensitive_features):
    base_model = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(
        base_model,
        constraints=DemographicParity()
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
    return mitigator
