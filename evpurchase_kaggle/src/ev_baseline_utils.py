"""
ev_baseline_utils.py

Baseline feature extraction and linear model utilities for the
Predicting Electric Vehicle Purchases competition.

Inspired by the modular style of base_utils_qwen.py, but simplified
for a tabular linear baseline.

Main components:
- EVFeatureExtractor
- LinearBaselineClassifier
- competition_score
- make_competition_scorer
- evaluate_holdout
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


# ============================================================
# HELPERS
# ============================================================

def _coerce_binary(value):
    """
    Convert Yes/No/True/False/0/1-like values into 0/1.
    Returns np.nan if unknown.
    """
    if value is None:
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return np.nan
        return 1 if float(value) >= 0.5 else 0

    s = str(value).strip().lower()

    if s in {"yes", "y", "true", "t", "1", "positive", "will_buy", "buy"}:
        return 1

    if s in {"no", "n", "false", "f", "0", "negative", "not_buy", "none"}:
        return 0

    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except Exception:
        return np.nan


def _coerce_range_anxiety(value):
    """
    Convert Range_Anxiety_Level into numeric:
        Low -> 0
        Medium -> 1
        High -> 2
    """
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return np.nan
        return float(value)

    s = str(value).strip().lower()

    mapping = {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
    }

    if s in mapping:
        return mapping[s]

    try:
        return float(s)
    except Exception:
        return np.nan


# ============================================================
# COMPETITION METRIC
# ============================================================

def competition_score(y_true, y_pred) -> float:
    """
    Official competition metric:

        ROC AUC between the predicted probability/risk score
        and the observed binary target Will_Buy_EV.

    This is intentionally a separate function so it can be reused
    in notebooks, scorers, and evaluation scripts.
    """
    y_true = np.asarray(y_true).ravel()
    y_true = pd.Series(y_true).map(_coerce_binary).fillna(0).astype(int).to_numpy()

    y_pred = np.asarray(y_pred)

    # If predict_proba output is given as shape (n, 2), take positive-class column.
    if y_pred.ndim == 2:
        if y_pred.shape[1] >= 2:
            y_pred = y_pred[:, 1]
        else:
            y_pred = y_pred.ravel()

    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

    # ROC AUC is undefined if only one class is present.
    if len(np.unique(y_true)) < 2:
        return 0.5

    return float(roc_auc_score(y_true, y_pred))


def make_competition_scorer(target_col: str = "Will_Buy_EV"):
    """
    Create an sklearn-compatible scorer for GridSearchCV / BayesSearchCV.
    """
    def _score(y_true, y_pred):
        return competition_score(y_true, y_pred)

    # Newer sklearn versions prefer response_method.
    # Older versions use needs_proba.
    try:
        return make_scorer(_score, response_method="predict_proba")
    except TypeError:
        return make_scorer(_score, needs_proba=True)


def evaluate_holdout(y_true, y_pred_proba, verbose: bool = True) -> dict:
    """
    Evaluate holdout predictions using the competition metric.
    """
    score = competition_score(y_true, y_pred_proba)

    if verbose:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        print(f"ROC AUC / Competition Score: {score:.4f}")

    return {
        "roc_auc": score,
        "competition_score": score,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class EVFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    Tabular feature extractor for EV purchase prediction.

    Parameters are deliberately exposed so they can be tuned separately
    from model parameters in GridSearchCV / BayesSearchCV.

    Feature sets:
        - univariate:
            Uses only one feature, e.g. Environmental_Concern_Level.

        - multivariate:
            Uses all available raw and optionally derived features.
    """

    DERIVED_FEATURES = [
        "Total_Charging_Stations",
        "Charging_Stations_per_10km",
        "Income_per_Car",
        "Log_Annual_Income_USD",
        "Concern_x_Total_Charging",
        "Concern_x_Log_Income",
        "High_Environmental_Concern",
        "Log_Daily_Commute_km",
    ]

    def __init__(
        self,
        feature_set: str = "multivariate",
        univariate_feature: str = "Environmental_Concern_Level",
        target_col: str = "Will_Buy_EV",
        drop_id: bool = True,
        impute_strategy: str = "median",
        scale_numeric: bool = True,
        onehot_categorical: bool = True,
        add_derived_features: bool = True,
    ):
        self.feature_set = feature_set
        self.univariate_feature = univariate_feature
        self.target_col = target_col
        self.drop_id = drop_id
        self.impute_strategy = impute_strategy
        self.scale_numeric = scale_numeric
        self.onehot_categorical = onehot_categorical
        self.add_derived_features = add_derived_features

    @staticmethod
    def _make_onehot_encoder():
        """
        Compatibility helper for different sklearn versions.
        """
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _prepare_dataframe(self, X) -> pd.DataFrame:
        """
        Convert input into a clean DataFrame and create derived features.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        df.columns = [str(c).strip() for c in df.columns]

        if self.drop_id and "id" in df.columns:
            df = df.drop(columns=["id"])

        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        # ------------------------------------------------------------
        # Known binary / ordinal conversions
        # ------------------------------------------------------------
        for col in ["Home_Charging_Possible", "Subsidy_Available"]:
            if col in df.columns:
                df[col] = df[col].map(_coerce_binary).astype("float")

        if "Range_Anxiety_Level" in df.columns:
            df["Range_Anxiety_Level"] = (
                df["Range_Anxiety_Level"]
                .map(_coerce_range_anxiety)
                .astype("float")
            )

        # ------------------------------------------------------------
        # Coerce expected numeric columns
        # ------------------------------------------------------------
        expected_numeric = [
            "Age",
            "Annual_Income_USD",
            "Daily_Commute_km",
            "Number_of_Cars_Owned",
            "Charging_Stations_Near_Home",
            "Charging_Stations_Near_Work",
            "Environmental_Concern_Level",
            "Home_Charging_Possible",
            "Subsidy_Available",
            "Range_Anxiety_Level",
        ]

        for col in expected_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ------------------------------------------------------------
        # Derived features
        # ------------------------------------------------------------
        home = df.get("Charging_Stations_Near_Home")
        work = df.get("Charging_Stations_Near_Work")

        if home is not None or work is not None:
            home_vals = home.fillna(0) if home is not None else 0
            work_vals = work.fillna(0) if work is not None else 0
            df["Total_Charging_Stations"] = home_vals + work_vals

        if "Total_Charging_Stations" in df.columns and "Daily_Commute_km" in df.columns:
            df["Charging_Stations_per_10km"] = (
                df["Total_Charging_Stations"]
                / ((df["Daily_Commute_km"].fillna(0) / 10.0) + 1.0)
            )

        if "Annual_Income_USD" in df.columns and "Number_of_Cars_Owned" in df.columns:
            df["Income_per_Car"] = (
                df["Annual_Income_USD"].fillna(0)
                / (df["Number_of_Cars_Owned"].fillna(0) + 1.0)
            )

        if "Annual_Income_USD" in df.columns:
            df["Log_Annual_Income_USD"] = np.log1p(
                df["Annual_Income_USD"].clip(lower=0)
            )

        if (
            "Environmental_Concern_Level" in df.columns
            and "Total_Charging_Stations" in df.columns
        ):
            df["Concern_x_Total_Charging"] = (
                df["Environmental_Concern_Level"].fillna(0)
                * df["Total_Charging_Stations"].fillna(0)
            )

        if (
            "Environmental_Concern_Level" in df.columns
            and "Log_Annual_Income_USD" in df.columns
        ):
            df["Concern_x_Log_Income"] = (
                df["Environmental_Concern_Level"].fillna(0)
                * df["Log_Annual_Income_USD"].fillna(0)
            )

        if "Environmental_Concern_Level" in df.columns:
            df["High_Environmental_Concern"] = (
                df["Environmental_Concern_Level"].fillna(0) >= 4
            ).astype(float)

        if "Daily_Commute_km" in df.columns:
            df["Log_Daily_Commute_km"] = np.log1p(
                df["Daily_Commute_km"].clip(lower=0)
            )

        return df

    def _select_features(self, df: pd.DataFrame):
        """
        Select feature columns based on feature_set.
        """
        if self.feature_set == "univariate":
            if self.univariate_feature in df.columns:
                return [self.univariate_feature]

            # Fallback if requested univariate feature is unavailable.
            fallback = [
                c
                for c in df.columns
                if c not in ["id", self.target_col]
            ]
            if not fallback:
                raise ValueError("No fallback features available for univariate mode.")
            return [fallback[0]]

        if self.feature_set == "multivariate":
            cols = [
                c
                for c in df.columns
                if c not in ["id", self.target_col]
            ]

            if not self.add_derived_features:
                cols = [c for c in cols if c not in self.DERIVED_FEATURES]

            return cols

        raise ValueError(
            f"Unknown feature_set: {self.feature_set}. "
            "Expected 'univariate' or 'multivariate'."
        )

    def fit(self, X, y=None):
        """
        Fit preprocessing pipeline.
        """
        X_df = self._prepare_dataframe(X)
        selected_features = self._select_features(X_df)

        self.selected_features_ = selected_features
        X_sel = X_df[selected_features].copy()

        num_cols = X_sel.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_sel.columns if c not in num_cols]

        self.numeric_features_ = num_cols
        self.categorical_features_ = cat_cols

        transformers = []

        if num_cols:
            num_steps = [
                ("imputer", SimpleImputer(strategy=self.impute_strategy)),
            ]

            if self.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))

            transformers.append(
                ("num", Pipeline(num_steps), num_cols)
            )

        if cat_cols:
            if self.onehot_categorical:
                cat_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", self._make_onehot_encoder()),
                    ]
                )
            else:
                cat_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                )

            transformers.append(
                ("cat", cat_pipeline, cat_cols)
            )

        if not transformers:
            raise ValueError("No features available to fit EVFeatureExtractor.")

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.preprocessor_.fit(X_sel, y)

        return self

    def transform(self, X):
        """
        Transform data using fitted preprocessing pipeline.
        """
        check_is_fitted(self, ["selected_features_", "preprocessor_"])

        X_df = self._prepare_dataframe(X)

        # Ensure selected columns exist, even if missing in new data.
        X_sel = pd.DataFrame(index=X_df.index)
        for col in self.selected_features_:
            if col in X_df.columns:
                X_sel[col] = X_df[col]
            else:
                X_sel[col] = np.nan

        transformed = self.preprocessor_.transform(X_sel)
        return np.asarray(transformed, dtype=float)


# ============================================================
# LINEAR BASELINE MODEL
# ============================================================

class LinearBaselineClassifier(ClassifierMixin, BaseEstimator):
    """
    Linear baseline wrapper.

    Supports:
        - linear_regression:
            Uses sklearn LinearRegression.
            The continuous prediction is treated as a risk score.
            For predict_proba, it is bounded using batch min-max scaling.

        - logistic_regression:
            Optional proper probabilistic linear classifier.
    """

    def __init__(
        self,
        model_type: str = "linear_regression",
        fit_intercept: bool = True,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 2000,
        random_state: int = 42,
        class_weight=None,
        clip_predictions: bool = True,
    ):
        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.clip_predictions = clip_predictions

    def _clean_X(self, X) -> np.ndarray:
        """
        Convert input to clean numeric matrix.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, X, y):
        """
        Fit linear model.
        """
        X = self._clean_X(X)

        y = np.asarray(y).ravel()
        y = pd.Series(y).map(_coerce_binary).fillna(0).astype(int).to_numpy()

        self.classes_ = np.unique(y)

        # Degenerate single-class case.
        if len(self.classes_) == 1:
            self.constant_class_ = int(self.classes_[0])
            self.model_ = None
            return self

        if self.model_type == "linear_regression":
            self.model_ = LinearRegression(
                fit_intercept=self.fit_intercept,
            )
            self.model_.fit(X, y)

        elif self.model_type == "logistic_regression":
            self.model_ = LogisticRegression(
                fit_intercept=self.fit_intercept,
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
            self.model_.fit(X, y)

        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Expected 'linear_regression' or 'logistic_regression'."
            )

        return self

    def predict_proba(self, X):
        """
        Return two-column probabilities/risk scores:
            column 0: score for class 0
            column 1: score for class 1

        For linear_regression, column 1 is a bounded risk score.
        """
        X = self._clean_X(X)
        n = X.shape[0]

        # Degenerate single-class case.
        if hasattr(self, "constant_class_"):
            p = np.full(n, float(self.constant_class_), dtype=float)
            return np.column_stack([1.0 - p, p])

        if self.model_type == "logistic_regression":
            proba = self.model_.predict_proba(X)
            classes = list(self.model_.classes_)

            if 1 in classes:
                positive_idx = classes.index(1)
            else:
                positive_idx = proba.shape[1] - 1

            p = proba[:, positive_idx]
            return np.column_stack([1.0 - p, p])

        # Linear regression risk score.
        raw = self.model_.predict(X)
        raw = np.asarray(raw, dtype=float)
        raw = np.nan_to_num(raw, nan=0.5, posinf=0.5, neginf=0.5)

        if self.clip_predictions:
            lo = np.min(raw)
            hi = np.max(raw)

            if hi - lo > 1e-12:
                p = (raw - lo) / (hi - lo)
            else:
                p = np.full_like(raw, 0.5)
        else:
            p = raw

        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        """
        Predict binary class.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)