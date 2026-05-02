from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WeightedSoftVotingClassifier:
    """Small pickle-safe soft-voting wrapper for manually fitted estimators."""

    estimators_: list[Any]
    weights: list[float]
    classes_: np.ndarray
    feature_names_in_: np.ndarray

    def predict_proba(self, X):
        total_weight = float(sum(self.weights))
        if total_weight <= 0:
            raise ValueError("Voting weights must sum to a positive value.")
        probs = None
        for estimator, weight in zip(self.estimators_, self.weights):
            p = estimator.predict_proba(X)
            probs = p * weight if probs is None else probs + (p * weight)
        return probs / total_weight

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
