"""Model factories, ONNX export and the ONNX scorer used by evaluate.py and predict()."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as LgbFloat
from onnxmltools.convert.xgboost.operator_converters.XGBoost import XGBConverter
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

from preprocess import FEATURES

MODELS = Path(__file__).resolve().parents[1] / "models"


def make(kind: str, params: dict):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, **params)
    # Fixed tree seeds keep the sweep comparisons repeatable.
    if kind == "xgboost":
        return xgb.XGBClassifier(random_state=0, n_jobs=8, eval_metric="logloss", **params)
    return lgb.LGBMClassifier(verbose=-1, random_state=0, n_jobs=8, **params)


def export(model, features: str, weight: float, threshold: float, path) -> None:
    """Write a fitted classifier as ONNX; features, blend weight and threshold go into metadata."""
    n = model.n_features_in_
    if isinstance(model, LogisticRegression):
        graph = convert_sklearn(model, initial_types=[("x", FloatTensorType([None, n]))],
                                options={id(model): {"zipmap": False}}, target_opset=15)
    elif isinstance(model, xgb.XGBClassifier):
        # onnxmltools 1.12 gives ONNX bools for an integer missing-direction field.
        add_node = XGBConverter._add_node
        def add_xgb_node(*args, **kwargs):
            kwargs["missing"] = int(kwargs["missing"])
            return add_node(*args, **kwargs)
        XGBConverter._add_node = staticmethod(add_xgb_node)
        try:
            graph = convert_xgboost(model, initial_types=[("x", LgbFloat([None, n]))],
                                    target_opset=15)
        finally:
            XGBConverter._add_node = staticmethod(add_node)
    else:
        graph = convert_lightgbm(model, initial_types=[("x", LgbFloat([None, n]))],
                                 zipmap=False, target_opset=15)
    # Scorer needs the feature recipe and blend settings alongside each graph.
    metadata = {"features": features, "weight": str(weight), "threshold": str(threshold)}
    if isinstance(model, xgb.XGBClassifier):
        metadata["zero_is_missing"] = "1"
    onnx.helper.set_model_props(graph, metadata)
    onnx.save(graph, path)


class Scorer:
    """URL strings in, phishing probabilities out: weighted sum of the ONNX files named
    models/<name>*.onnx. Feature code runs in Python, the classifiers in onnxruntime."""

    def __init__(self, name: str):
        self.parts = []
        for path in sorted(MODELS.glob(f"{name}*.onnx")):
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            meta = session.get_modelmeta().custom_metadata_map
            outputs = [output.name for output in session.get_outputs()]
            # Exporters name probability outputs differently; select by meaning.
            probability = next((output for output in outputs
                                if "probab" in output.lower()), outputs[-1])
            # Keep each component's weight and feature-specific missing behavior together.
            self.parts.append((session, float(meta["weight"]),
                               meta.get("zero_is_missing") == "1", probability))
        self.features, self.threshold = FEATURES[meta["features"]], float(meta["threshold"])  # Components share this setup.

    def predict_proba(self, urls, batch: int = 256) -> np.ndarray:
        # Small batches keep dense ONNX inputs from using too much memory.
        urls, out = list(urls), []
        for i in range(0, len(urls), batch):
            x = self.features(urls[i:i + batch]).toarray()
            parts = []
            for session, weight, zero_missing, output in self.parts:
                # XGBoost saw absent sparse entries as missing; ONNX gets dense zeros.
                values = np.where(x == 0, np.nan, x) if zero_missing else x
                parts.append(weight * session.run([output], {"x": values})[0][:, 1])  # Sum the phishing class.
            out.append(sum(parts))
        return np.clip(np.concatenate(out), 0.0, 1.0).astype(float)
