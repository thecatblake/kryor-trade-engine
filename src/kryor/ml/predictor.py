"""ML inference — load trained LightGBM model and produce signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kryor.ml.trainer import load_model


class MLPredictor:
    """Loads a trained model and provides prediction interface."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        bundle = load_model(self.model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.metrics = bundle["metrics"]
        self.trained_at = bundle["trained_at"]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Returns array of shape (n_samples, 3) with [P(SELL), P(HOLD), P(BUY)]."""
        X = features[self.feature_cols].values
        return self.model.predict_proba(X)

    def predict_signal(self, features_row: pd.Series, buy_threshold: float = 0.5,
                       sell_threshold: float = 0.5) -> tuple[str, float]:
        """Predict signal for single bar.

        Returns:
            ("buy", confidence) | ("sell", confidence) | ("hold", confidence)
        """
        X = features_row[self.feature_cols].values.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]  # [p_sell, p_hold, p_buy]
        p_sell, p_hold, p_buy = proba

        if p_buy > buy_threshold and p_buy > p_sell:
            return "buy", float(p_buy)
        if p_sell > sell_threshold and p_sell > p_buy:
            return "sell", float(p_sell)
        return "hold", float(p_hold)

    def feature_importance(self) -> dict[str, float]:
        importance = self.model.feature_importances_
        return dict(zip(self.feature_cols, importance.tolist()))
