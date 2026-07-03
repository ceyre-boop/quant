"""D3 — the model zoo, small by design: regularized logistic, gradient-boosted trees,
small MLP. Every member emits CALIBRATED probabilities and ships inside an
abstain-below-confidence wrapper — RISK_CONSTITUTION Article 4 ("doing nothing is a
position"), encoded. No torch/lightgbm: sklearn + xgboost only (present in env)."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str):
    """Fresh calibrated estimator. kind ∈ {logistic, xgb, mlp}."""
    if kind == "logistic":
        base = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    elif kind == "xgb":
        from xgboost import XGBClassifier
        base = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    elif kind == "mlp":
        base = make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-3,
                                           max_iter=3000, random_state=42))
    else:
        raise ValueError(f"unknown zoo member {kind!r} — the zoo is small by design")
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


ZOO = ("logistic", "xgb", "mlp")


class AbstainingModel:
    """Article-4 wrapper: predictions below the confidence floor become ABSTAIN (None)."""

    def __init__(self, model, min_confidence: float = 0.60):
        self.model = model
        self.min_confidence = float(min_confidence)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decide(self, X) -> list[dict]:
        """Per row: {p, decision ∈ {LONG, SHORT, ABSTAIN}, confidence}."""
        out = []
        for p in self.model.predict_proba(X)[:, 1]:
            conf = max(p, 1 - p)
            decision = "ABSTAIN" if conf < self.min_confidence else ("LONG" if p > 0.5 else "SHORT")
            out.append({"p": float(p), "confidence": float(conf), "decision": decision})
        return out
