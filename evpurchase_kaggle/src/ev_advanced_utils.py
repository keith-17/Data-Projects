"""
ev_advanced_utils.py
Advanced extraction: screenshot feats + domain interactions + recipe + digits
+ EDA-rule features + configurable categorical encoding strategies.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from ev_baseline_utils import EVFeatureExtractor


def _first(df, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _num(series, index):
    if series is None:
        return pd.Series(0.0, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


_ORD_MAP = [("very", 4), ("extreme", 4), ("severe", 4), ("high", 3),
            ("moderate", 2), ("med", 2), ("mid", 2), ("low", 1), ("none", 0)]


def _ordinal(series, index):
    if series is None:
        return pd.Series(0.0, index=index)
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().mean() < 0.5:
        low = series.astype(str).str.lower()
        v = pd.to_numeric(low.map(
            lambda t: next((vv for k, vv in _ORD_MAP if k in t), np.nan)),
            errors="coerce")
    return v.fillna(v.median() if v.notna().any() else 0.0)


def _bin(series, index):
    if series is None:
        return pd.Series(0.0, index=index)
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().mean() >= 0.5:
        return v.clip(0, 1).fillna(0.0)
    low = series.astype(str).str.strip().str.lower()
    return low.map(lambda t: 1.0 if t.startswith(("y", "t", "1"))
                   else 0.0 if t.startswith(("n", "f", "0"))
                   else np.nan).fillna(0.0)


def _urban_flag(city, index):
    if city is None:
        return pd.Series(0.0, index=index)
    if pd.api.types.is_numeric_dtype(city):
        return (pd.to_numeric(city, errors="coerce") == 0).fillna(False).astype(float)
    s = city.astype(str).str.strip().str.lower()
    return (s.str.contains("urban", regex=False, na=False)
            & ~s.str.startswith(("non", "not"))).astype(float)


class TripleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothings=(1.0, 10.0, 100.0)):
        self.smoothings = smoothings

    def fit(self, X, y=None):
        X = pd.DataFrame(X); yv = pd.Series(np.asarray(y, float), index=X.index)
        self.global_mean_ = float(yv.mean()); self.columns_ = list(X.columns)
        self.maps_ = {}
        for c in self.columns_:
            key = X[c].astype(str); cnt = yv.groupby(key).count(); mean = yv.groupby(key).mean()
            self.maps_[c] = [(cnt * mean + m * self.global_mean_) / (cnt + m)
                             for m in self.smoothings]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = [key_map for c in self.columns_
               for key_map in [X[c].astype(str).map(m).fillna(self.global_mean_).to_numpy(float)
                               for m in self.maps_[c]]]
        return np.column_stack(out) if out else np.zeros((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_te{m:g}" for c in self.columns_
                           for m in self.smoothings], dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X); self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].astype(str).value_counts(normalize=True) for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = [X[c].astype(str).map(self.maps_[c]).fillna(0.0).to_numpy(float)
               for c in self.columns_]
        return np.column_stack(out) if out else np.zeros((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_freq" for c in self.columns_], dtype=object)


class AdvancedEVFeatureExtractor(EVFeatureExtractor):
    SCREENSHOT = ["Income_Age_Ratio", "Commute_x_Charging", "Commute_x_Anxiety",
                  "Income_x_Subsidy", "Cars_per_Income", "Is_Young_Urban",
                  "Charging_Density", "Concern_x_Anxiety"]
    DOMAIN = ["Low_Income_x_Subsidy", "Concern_x_Subsidy", "CityType_x_HomeCharging"]
    RECIPE = ["Medium_Range_Anxiety", "High_Range_Anxiety", "Buy_Score"]
    EDA_RULES = ["No_Nearby_Charging", "Commute_Is_5km", "Commute_GE_83",
                 "Income_GT_169972", "Income_Dead_Band",
                 "Subsidy_x_HomeCharging", "Anxiety_Concern_Cross"]
    DEFAULT_ORDINAL = ("Range_Anxiety_Level", "Environmental_Concern_Level")

    def __init__(self, feature_set="multivariate",
                 univariate_feature="Environmental_Concern_Level",
                 target_col="Will_Buy_EV", drop_id=True,
                 impute_strategy="median", scale_numeric=False,
                 onehot_categorical=True, add_derived_features=True,
                 add_advanced_features=True, add_domain_features=True,
                 add_recipe_features=True, add_eda_rules=True,
                 add_digit_features=True,
                 categorical_strategy="auto", strategy_overrides=None,
                 ordinal_maps=None, onehot_min_frequency=None,
                 onehot_max_categories=None, high_card_threshold=8,
                 extra_encodings=True,
                 low_income_threshold=31004.0, young_age_threshold=35,
                 digit_columns=("Annual_Income_USD", "Age", "Daily_Commute_km"),
                 digit_exponents=(1, 0, -1, -2, -3, -4)):
        super().__init__(feature_set=feature_set, univariate_feature=univariate_feature,
                         target_col=target_col, drop_id=drop_id,
                         impute_strategy=impute_strategy, scale_numeric=scale_numeric,
                         onehot_categorical=onehot_categorical,
                         add_derived_features=add_derived_features)
        self.add_advanced_features = add_advanced_features
        self.add_domain_features = add_domain_features
        self.add_recipe_features = add_recipe_features
        self.add_eda_rules = add_eda_rules
        self.add_digit_features = add_digit_features
        self.categorical_strategy = categorical_strategy
        self.strategy_overrides = strategy_overrides
        self.ordinal_maps = ordinal_maps
        self.onehot_min_frequency = onehot_min_frequency
        self.onehot_max_categories = onehot_max_categories
        self.high_card_threshold = high_card_threshold
        self.extra_encodings = extra_encodings
        self.low_income_threshold = low_income_threshold
        self.young_age_threshold = young_age_threshold
        self.digit_columns = digit_columns
        self.digit_exponents = digit_exponents

        digits = [f"{c}_d{e}" for c in digit_columns for e in digit_exponents]
        self.digit_features_ = digits
        self.feature_groups_ = {
            "base": list(EVFeatureExtractor.DERIVED_FEATURES),
            "screenshot": list(self.SCREENSHOT),
            "domain": list(self.DOMAIN),
            "recipe": list(self.RECIPE),
            "eda_rules": list(self.EDA_RULES),
            "digits": digits,
        }
        derived = list(EVFeatureExtractor.DERIVED_FEATURES)
        if add_advanced_features: derived += self.SCREENSHOT
        if add_domain_features:   derived += self.DOMAIN
        if add_recipe_features:   derived += self.RECIPE
        if add_eda_rules:         derived += self.EDA_RULES
        if add_digit_features:    derived += digits
        self.DERIVED_FEATURES = derived

    # ----------------------------------------------------------
    def _prepare_dataframe(self, X):
        df = super()._prepare_dataframe(X)
        idx = df.index
        income = _num(_first(df, ["Annual_Income_USD", "Income"]), idx)
        age = _num(_first(df, ["Age"]), idx)
        commute = _num(_first(df, ["Daily_Commute_km"]), idx)
        cars = _num(_first(df, ["Number_of_Cars_Owned", "Number_of_Cars"]), idx)
        subsidy = _bin(_first(df, ["Subsidy_Available"]), idx)
        home = _bin(_first(df, ["Home_Charging_Possible"]), idx)
        anxiety = _ordinal(_first(df, ["Range_Anxiety_Level"]), idx)
        concern = _ordinal(_first(df, ["Environmental_Concern_Level"]), idx)
        ch_home = _num(_first(df, ["Charging_Stations_Near_Home"]), idx)
        ch_work = _num(_first(df, ["Charging_Stations_Near_Work"]), idx)
        total = _num(_first(df, ["Total_Charging_Stations"]), idx)
        urban = _urban_flag(_first(df, ["City_Type", "City"]), idx)

        df["Income_Age_Ratio"] = income / (age + 1.0)
        df["Commute_x_Charging"] = commute * total
        df["Commute_x_Anxiety"] = commute * anxiety
        df["Income_x_Subsidy"] = income * subsidy
        df["Cars_per_Income"] = cars / income.clip(lower=1.0)
        df["Is_Young_Urban"] = (age < self.young_age_threshold).astype(float) * urban
        df["Charging_Density"] = total / (commute + 1.0)
        df["Concern_x_Anxiety"] = concern * anxiety
        df["Low_Income_x_Subsidy"] = (income < self.low_income_threshold).astype(float) * subsidy
        df["Concern_x_Subsidy"] = concern * subsidy
        city_s = _first(df, ["City_Type", "City"])
        city_s = city_s.astype(str) if city_s is not None else pd.Series("missing", index=idx)
        df["CityType_x_HomeCharging"] = city_s.fillna("missing") + "_chg_" + \
            home.map({0.0: "no", 1.0: "yes"}).astype(str)
        med = (anxiety == 2).astype(float); high = (anxiety >= 3).astype(float)
        df["Medium_Range_Anxiety"] = med
        df["High_Range_Anxiety"] = high
        df["Buy_Score"] = (1.2 * (income / 100000.0) + 0.6 * concern
                           + 2.0 * subsidy - 1.0 * med - 3.0 * high)
        df["No_Nearby_Charging"] = ((ch_home == 0) & (ch_work == 0)).astype(float)
        df["Commute_Is_5km"] = (commute == 5).astype(float)
        df["Commute_GE_83"] = (commute >= 83).astype(float)
        df["Income_GT_169972"] = (income > 169972).astype(float)
        df["Income_Dead_Band"] = income.between(31004, 41970).astype(float)
        df["Subsidy_x_HomeCharging"] = subsidy * home
        df["Anxiety_Concern_Cross"] = (anxiety.astype(int).astype(str) + "_"
                                       + concern.astype(int).astype(str))
        if self.add_digit_features:
            for c in self.digit_columns:
                src = _first(df, [c])
                if src is None:
                    continue
                v = pd.to_numeric(src, errors="coerce").fillna(0.0).abs()
                for e in self.digit_exponents:
                    df[f"{c}_d{e}"] = np.floor(v / (10.0 ** e)) % 10
        return df

    # ----------------------------------------------------------
    def _resolve_strategies(self, X_sel):
        out = {}
        for c in [c for c in X_sel.columns if not pd.api.types.is_numeric_dtype(X_sel[c])]:
            s = (self.strategy_overrides or {}).get(c, self.categorical_strategy)
            n_un = X_sel[c].nunique(dropna=True)
            if s == "auto":
                if n_un <= 2:
                    s = "binary"
                elif c in self.DEFAULT_ORDINAL or (self.ordinal_maps and c in self.ordinal_maps):
                    s = "ordinal"
                elif n_un > self.high_card_threshold:
                    s = "target"
                else:
                    s = "onehot_drop_first"
            out[c] = s
        return out

    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X_df = self._prepare_dataframe(X)
        self.selected_features_ = self._select_features(X_df)
        X_sel = X_df[self.selected_features_].copy()

        strat = self._resolve_strategies(X_sel)
        for c, s in strat.items():
            if s == "binary":
                X_sel[c] = _bin(X_sel[c], X_sel.index)
        num_cols = [c for c in X_sel.columns
                    if c in [k for k, v in strat.items() if v == "binary"]
                    or pd.api.types.is_numeric_dtype(X_sel[c])]
        cat_cols = [c for c in X_sel.columns if c not in num_cols]
        self.numeric_features_, self.categorical_features_ = num_cols, cat_cols

        transformers = []
        if num_cols:
            steps = [("imputer", SimpleImputer(strategy=self.impute_strategy))]
            if self.scale_numeric:
                steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(steps), num_cols))

        groups = {s: [c for c in cat_cols if strat.get(c) == s]
                  for s in ["onehot", "onehot_drop_first", "ordinal", "target", "freq"]}
        if groups["onehot"]:
            transformers.append(("onehot", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore",
                                         min_frequency=self.onehot_min_frequency,
                                         max_categories=self.onehot_max_categories))]),
                groups["onehot"]))
        if groups["onehot_drop_first"]:
            transformers.append(("onehot_df", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                         min_frequency=self.onehot_min_frequency,
                                         max_categories=self.onehot_max_categories))]),
                groups["onehot_drop_first"]))
        if groups["ordinal"]:
            cats = [list(self.ordinal_maps[c]) if (self.ordinal_maps and c in self.ordinal_maps)
                    else sorted(X_sel[c].dropna().unique()) for c in groups["ordinal"]]
            transformers.append(("ord", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value",
                                           unknown_value=-1))]),
                groups["ordinal"]))
        if groups["target"] and y is not None:
            transformers.append(("tgt", TripleTargetEncoder(), groups["target"]))
        elif groups["target"]:
            transformers.append(("tgt_f", FrequencyEncoder(), groups["target"]))
        if groups["freq"]:
            transformers.append(("freq", FrequencyEncoder(), groups["freq"]))
        if self.extra_encodings:
            hi = [c for c in cat_cols if X_sel[c].nunique() > self.high_card_threshold
                  and strat.get(c) not in ("target", "freq")]
            if hi:
                transformers.append(("freq_extra", FrequencyEncoder(), hi))
        if not transformers:
            raise ValueError("No features available to fit AdvancedEVFeatureExtractor.")
        self.preprocessor_ = ColumnTransformer(transformers, remainder="drop")
        self.preprocessor_.fit(X_sel, y)
        return self

    # ----------------------------------------------------------
    def transform_dataframe(self, X):
        check_is_fitted(self, ["selected_features_", "preprocessor_"])
        df = self._prepare_dataframe(X)
        for c in self.selected_features_:
            if c not in df.columns:
                df[c] = np.nan
        return df[self.selected_features_].copy()