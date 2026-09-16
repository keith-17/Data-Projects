"""
ev_analysis_utils.py
EDA-inspired signal ranking, correlation/interaction studio, drift,
derived-vs-original audit, prediction diagnostics, EDA segment auditor.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, brier_score_loss,
                             confusion_matrix)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


def _coerce_num(s):
    return pd.to_numeric(s, errors="coerce")


def _cramers_v(cat, y):
    tab = pd.crosstab(cat, y)
    n = tab.values.sum()
    exp = np.outer(tab.sum(axis=1).values, tab.sum(axis=0).values) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((tab.values - exp) ** 2 / np.where(exp > 0, exp, 1)))
    denom = min(tab.shape[0] - 1, tab.shape[1] - 1)
    return float(np.sqrt(chi2 / n / denom)) if denom > 0 else 0.0


def _information_value(cat, y, n_bins=10):
    y = np.asarray(y, dtype=int)
    s = _coerce_num(cat)
    if s.notna().mean() > 0.5 and s.nunique() > n_bins:
        b = pd.qcut(s, q=n_bins, duplicates="drop")
    else:
        b = cat.astype(str)
    df = pd.DataFrame({"b": b.astype(str), "y": y})
    g = df.groupby("b")["y"].agg(["sum", "count"])
    g["ne"] = g["count"] - g["sum"]
    eps = 1e-4
    de = (g["sum"] + eps) / (g["sum"].sum() + eps * len(g))
    dn = (g["ne"] + eps) / (g["ne"].sum() + eps * len(g))
    woe = np.log(de / dn)
    return float(((de - dn) * woe).sum())


def _mutual_information(values, y, random_state):
    """Return MI when sklearn's optional native backend is available.

    Some Windows BLAS builds fail while ``mutual_info_classif`` queries the
    thread-pool metadata.  Signal ranking is a diagnostic, so preserve the
    rest of the table instead of failing the entire notebook in that case.
    """
    try:
        return float(mutual_info_classif(values, y, random_state=random_state)[0])
    except (OSError, ValueError):
        return np.nan


class SignalRanker:
    """Univariate signal table: AUC / Cramer's V / IV / MI (EDA-notebook style)."""

    def __init__(self, y):
        # Keep the index when ``y`` is a Series.  Training splits retain the
        # source-row index, so dropping it makes sampled features and labels
        # point at different observations.
        self.y = pd.Series(y, copy=True).astype(int)

    def rank(self, X_df, n_sample=60000, random_state=42):
        n = min(n_sample, len(X_df))
        df = X_df.sample(n=n, random_state=random_state)
        if df.index.isin(self.y.index).all():
            y = self.y.reindex(df.index).to_numpy(dtype=int)
        elif len(self.y) == len(X_df) and X_df.index.is_unique:
            # Array-like targets have a default RangeIndex; align them by
            # the sampled row positions as a safe fallback.
            positions = X_df.index.get_indexer(df.index)
            y = self.y.to_numpy(dtype=int)[positions]
        else:
            raise ValueError("Target values cannot be aligned with X_df rows.")
        rows = []
        for c in df.columns:
            s = df[c]
            num = _coerce_num(s)
            if s.dtype == object or num.notna().mean() < 0.5:
                v = s.astype(str)
                aucs = [max(roc_auc_score(y, (v == lv).astype(int)),
                            1 - roc_auc_score(y, (v == lv).astype(int)))
                        for lv in v.value_counts().index[:6]]
                rows.append({"feature": c, "kind": "cat",
                             "auc": max(aucs) if aucs else 0.5,
                             "cramers_v": _cramers_v(v, y),
                             "iv": _information_value(v, y),
                             "mi": _mutual_information(
                                 v.astype("category").cat.codes.to_numpy().reshape(-1, 1),
                                 y, random_state)})
            else:
                v = num.fillna(num.median())
                a = roc_auc_score(y, v)
                rows.append({"feature": c, "kind": "num",
                             "auc": max(a, 1 - a),
                             "cramers_v": _cramers_v(
                                 pd.qcut(v, q=10, duplicates="drop").astype(str), y),
                             "iv": _information_value(v, y),
                             "mi": _mutual_information(
                                 v.to_numpy().reshape(-1, 1), y, random_state)})
        self.table_ = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
        return self.table_

    def plot(self, top=15, metric="auc"):
        t = self.table_.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(t.feature, t[metric],
                color=["tab:orange" if k == "cat" else "tab:blue" for k in t.kind])
        ax.set_title(f"Univariate signal ranking (top {top}, metric={metric})")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout(); plt.show()
        return fig


class CorrelationStudio:
    def __init__(self, X_df, y=None):
        self.X = X_df; self.y = None if y is None else np.asarray(y, dtype=int)

    def numeric_frame(self):
        out = self.X.apply(_coerce_num)
        return out.dropna(axis=1, how="all")

    def corr_matrix(self, top_k=18, method="spearman"):
        M = self.numeric_frame()
        if self.y is not None:
            tgt = M.corrwith(pd.Series(self.y, index=M.index), method=method).abs()
            cols = tgt.sort_values(ascending=False).head(top_k).index.tolist()
        else:
            cols = M.nunique().sort_values(ascending=False).head(top_k).index.tolist()
        return M[cols].corr(method=method)

    def plot_heatmap(self, top_k=18, method="spearman"):
        C = self.corr_matrix(top_k=top_k, method=method)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(C)), C.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(C)), C.columns, fontsize=8)
        for i in range(len(C)):
            for j in range(len(C)):
                ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{method.capitalize()} correlation (top-{top_k} by |corr| with target)")
        plt.tight_layout(); plt.show()
        return fig

    def interaction_pivot(self, row_col, col_col, y):
        df = pd.DataFrame({"r": self.X[row_col].astype(str),
                           "c": self.X[col_col].astype(str),
                           "y": np.asarray(y, dtype=int)})
        piv = df.pivot_table(index="r", columns="c", values="y", aggfunc="mean")
        cnt = df.pivot_table(index="r", columns="c", values="y", aggfunc="size")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(piv.shape[0]), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                n = cnt.values[i, j]
                ax.text(j, i, f"{piv.values[i, j]:.1%}\nn={int(n):,}",
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Target rate: {row_col} × {col_col} (Simpson check)")
        plt.tight_layout(); plt.show()
        return piv

    def psi(self, train_s, test_s):
        tr, te = _coerce_num(train_s), _coerce_num(test_s)
        if tr.notna().mean() < 0.5:
            tr, te = train_s.astype(str), test_s.astype(str)
            cats = tr.value_counts(normalize=True)
            p = cats.reindex(cats.index.union(te.value_counts(normalize=True).index)).fillna(1e-4)
            q = te.value_counts(normalize=True).reindex(p.index).fillna(1e-4)
        else:
            edges = np.unique(np.quantile(tr.dropna(), np.linspace(0, 1, 11)))
            pb = pd.cut(tr, bins=edges, include_lowest=True).value_counts(normalize=True)
            qb = pd.cut(te, bins=edges, include_lowest=True).value_counts(normalize=True)
            p = pb.reindex(pb.index.union(qb.index)).fillna(1e-4)
            q = qb.reindex(p.index).fillna(1e-4)
        p, q = p.clip(lower=1e-4), q.clip(lower=1e-4)
        return float(((p - q) * np.log(p / q)).sum())

    def drift_table(self, X_train_df, X_test_df):
        rows = [{"feature": c, "psi": self.psi(X_train_df[c], X_test_df[c])}
                for c in X_train_df.columns if c in X_test_df.columns]
        t = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        t["flag"] = np.where(t.psi > 0.2, "HIGH", np.where(t.psi > 0.1, "watch", "ok"))
        return t


class DerivedVsOriginalAudit:
    def importance_table(self, pipeline):
        ext = pipeline.named_steps["extractor"]
        sel = pipeline.named_steps["selector"]
        mdl = pipeline.named_steps["model"]
        enc = list(ext.preprocessor_.get_feature_names_out())
        names = [n for n, m in zip(enc, sel.get_support()) if m]
        raw = list(ext.selected_features_)
        pat = "(" + "|".join(sorted(raw, key=len, reverse=True)) + ")"
        df = pd.DataFrame({"encoded": names, "importance": mdl.feature_importances_})
        df["root"] = (df.encoded.str.split("__").str[-1]
                      .str.extract(pat, expand=False).fillna(df.encoded))
        grp_of = {f: g for g, fs in ext.feature_groups_.items() for f in fs}
        df["group"] = df.root.map(grp_of).fillna("original")
        df["origin"] = np.where(df.group == "original", "original", "derived")
        return (df.groupby(["root", "group", "origin"], as_index=False)
                  .agg(importance=("importance", "sum"),
                       encoded_cols=("encoded", "size"))
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    def summary(self, imp_tbl):
        s = imp_tbl.groupby("origin")["importance"].sum()
        w = imp_tbl.groupby("group")[["importance", "encoded_cols"]].sum()
        w["importance_per_col"] = w.importance / w.encoded_cols.clip(lower=1)
        return (f"derived share of importance = {s.get('derived', 0):.1%} | "
                f"original share = {s.get('original', 0):.1%}\n"
                + w.sort_values("importance_per_col", ascending=False).to_string())

    def run_ablation(self, base_pipeline, X_tr, y_tr, X_te, y_te):
        configs = [("ALL", {}),
                   ("ORIGINAL_ONLY", {"extractor__add_derived_features": False}),
                   ("NO_SCREENSHOT", {"extractor__add_advanced_features": False}),
                   ("NO_DOMAIN", {"extractor__add_domain_features": False}),
                   ("NO_RECIPE", {"extractor__add_recipe_features": False}),
                   ("NO_EDA_RULES", {"extractor__add_eda_rules": False}),
                   ("NO_DIGITS", {"extractor__add_digit_features": False})]
        rows = []
        for label, params in configs:
            p = clone(base_pipeline)
            p.set_params(model__n_estimators=60, **params)
            p.fit(X_tr, y_tr)
            rows.append({"config": label,
                         "holdout_auc": roc_auc_score(y_te, p.predict_proba(X_te)[:, 1])})
        t = pd.DataFrame(rows)
        base = t.loc[t.config == "ALL", "holdout_auc"].iloc[0]
        t["delta_vs_all"] = t.holdout_auc - base
        return t

    def plot(self, imp_tbl, ablation_df):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        top = imp_tbl.head(15).iloc[::-1]
        axes[0].barh(top.root + " [" + top.group + "]", top.importance,
                     color=["seagreen" if o == "derived" else "slategray" for o in top.origin])
        axes[0].set_title("Top-15 roots (green = derived)")
        a = ablation_df.iloc[::-1]
        axes[1].barh(a.config, a.holdout_auc,
                     color=["crimson" if d < -0.002 else "steelblue" for d in a.delta_vs_all])
        axes[1].set_title("Ablation: holdout AUC per feature family")
        for ax in axes:
            ax.grid(alpha=0.3, axis="x")
        plt.tight_layout(); plt.show()
        return fig


class PredictionDiagnostics:
    def __init__(self, y_true, proba, name="RF"):
        self.y = np.asarray(y_true, dtype=int)
        self.p = np.asarray(proba, dtype=float)
        self.name = name

    def metrics_table(self):
        fpr, tpr, _ = roc_curve(self.y, self.p)
        return pd.DataFrame([{
            "model": self.name,
            "roc_auc": roc_auc_score(self.y, self.p),
            "pr_auc": average_precision_score(self.y, self.p),
            "brier": brier_score_loss(self.y, self.p),
            "acc@0.5": (self.p >= 0.5).astype(int).eq(self.y).mean(),
            "youden_thr": np.linspace(0, 1, 101)[np.argmax(tpr - fpr)],
        }])

    def calibration_table(self, n_bins=10):
        b = pd.qcut(self.p, q=n_bins, duplicates="drop")
        return (pd.DataFrame({"bin": b, "y": self.y, "p": self.p})
                .groupby("bin", observed=True)
                .agg(mean_pred=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
                .reset_index())

    def threshold_sweep(self, n=101):
        th = np.linspace(0, 1, n)
        rows = []
        for t in th:
            pr = (self.p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y, pr, labels=[0, 1]).ravel()
            rows.append({"thr": t,
                         "acc": (tp + tn) / (tp + tn + fp + fn),
                         "precision": tp / max(tp + fp, 1),
                         "recall": tp / max(tp + fn, 1),
                         "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                         "youden": tp / max(tp + fn, 1) - fp / max(fp + tn, 1)})
        return pd.DataFrame(rows)

    def best_threshold(self, objective="youden"):
        sw = self.threshold_sweep()
        return float(sw.loc[sw[objective].idxmax(), "thr"])

    def lift_table(self, n_bins=10):
        df = pd.DataFrame({"p": self.p, "y": self.y})
        df["decile"] = pd.qcut(df.p, q=n_bins, labels=False, duplicates="drop")
        g = df.groupby("decile").agg(n=("y", "size"), rate=("y", "mean"), mean_p=("p", "mean"))
        g["lift"] = g.rate / max(self.y.mean(), 1e-9)
        return g.iloc[::-1]

    def error_anatomy(self, X_df, columns):
        pr = (self.p >= 0.5).astype(int)
        kind = np.where(pr == 1, np.where(self.y == 1, "TP", "FP"),
                        np.where(self.y == 1, "FN", "TN"))
        out = {}
        for c in columns:
            v = _coerce_num(X_df[c])
            if v is None or v.notna().mean() < 0.5:
                v = X_df[c].astype("category").cat.codes.replace(-1, np.nan)
            out[c] = pd.Series(v, index=X_df.index).groupby(kind).mean()
        anat = pd.DataFrame(out).T
        anat["overall"] = anat.mean(axis=1)
        return anat

    def plot_dashboard(self, X_df=None, anatomy_cols=None):
        fig = plt.figure(figsize=(19, 10))
        gs = fig.add_gridspec(2, 3)
        cal = self.calibration_table()
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(cal.mean_pred, cal.observed_rate, "o-", color="tab:blue", label="model")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("Calibration (reliability)"); ax.legend(); ax.grid(alpha=0.3)
        sw = self.threshold_sweep()
        ax = fig.add_subplot(gs[0, 1])
        for m, c in [("precision", "tab:green"), ("recall", "tab:red"), ("f1", "tab:orange")]:
            ax.plot(sw.thr, sw[m], lw=1.5, label=m, color=c)
        ax.axvline(self.best_threshold(), color="black", ls=":", lw=1)
        ax.set_title(f"Threshold sweep (youden thr={self.best_threshold():.2f})")
        ax.legend(); ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[0, 2])
        lt = self.lift_table()
        ax.bar(lt.index.astype(str), lt.lift, color="mediumpurple")
        ax.axhline(1, color="gray", ls="--")
        ax.set_title("Lift by score decile (top→bottom)")
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(self.p[self.y == 0], bins=50, alpha=0.6, color="tab:red", density=True, label="true 0")
        ax.hist(self.p[self.y == 1], bins=50, alpha=0.6, color="tab:green", density=True, label="true 1")
        ax.legend(); ax.set_title("Score separation")
        if X_df is not None and anatomy_cols:
            anat = self.error_anatomy(X_df.reset_index(drop=True), anatomy_cols)
            num = anat.drop(columns=["overall"], errors="ignore")
            z = (num.sub(num["TN"], axis=0)
                 .div(num["TN"].replace(0, np.nan).abs().add(1e-9), axis=0))
            ax = fig.add_subplot(gs[1, 1:])
            z[["FN", "FP"]].plot.barh(ax=ax, color={"FN": "crimson", "FP": "goldenrod"})
            ax.set_title("Error anatomy: how FN (missed buyers) & FP differ from TN (z-scale)")
        else:
            ax = fig.add_subplot(gs[1, 1:]); ax.axis("off")
        fig.suptitle(f"{self.name} prediction diagnostics", fontweight="bold")
        fig.tight_layout(); plt.show()
        return fig


class EDASegmentAuditor:
    """Does the model know what the EDA knows? (mirrors 'Signal that Matters')."""

    def segments(self, X_df):
        income = _coerce_num(X_df.get("Annual_Income_USD"))
        commute = _coerce_num(X_df.get("Daily_Commute_km"))
        ch_h = _coerce_num(X_df.get("Charging_Stations_Near_Home")).fillna(0)
        ch_w = _coerce_num(X_df.get("Charging_Stations_Near_Work")).fillna(0)
        sub = _coerce_num(X_df.get("Subsidy_Available"))
        if sub is None or sub.notna().mean() < 0.5:
            sub = X_df["Subsidy_Available"].astype(str).str.lower().str.startswith(("y", "t", "1")).astype(float)
        home = _coerce_num(X_df.get("Home_Charging_Possible"))
        if home is None or home.notna().mean() < 0.5:
            home = X_df["Home_Charging_Possible"].astype(str).str.lower().str.startswith(("y", "t", "1")).astype(float)
        anx = X_df.get("Range_Anxiety_Level")
        anx_s = anx.astype(str).str.lower() if anx is not None else None
        anx_v = _coerce_num(anx)
        high = (anx_v >= 3) if (anx_v is not None and anx_v.notna().mean() > 0.5) \
            else anx_s.str.contains("high")
        med = (anx_v == 2) if (anx_v is not None and anx_v.notna().mean() > 0.5) \
            else anx_s.str.contains("med")
        con = _coerce_num(X_df.get("Environmental_Concern_Level"))
        return {
            "zero_nearby_charging": (ch_h == 0) & (ch_w == 0),
            "commute_5km_spike": commute == 5,
            "commute_ge_83 (0 buyers in EDA)": commute >= 83,
            "income_gt_169972 (100% buyers)": income > 169972,
            "income_dead_band_31k_42k (0 buyers)": income.between(31004, 41970),
            "income_lt_31004": income < 31004,
            "no_subsidy_no_homechg (0.4%)": (sub == 0) & (home == 0),
            "anxiety_high (0.14% buy)": high.astype(bool),
            "anxiety_medium": med.astype(bool),
            "concern_5 (coin flip)": con == 5,
            "low_anx_concern_5 (54.5%)": (~high.astype(bool)) & (~med.astype(bool)) & (con == 5),
        }

    def audit(self, X_df, y, proba):
        y = np.asarray(y, dtype=int); proba = np.asarray(proba, float)
        rows = []
        for label, mask in self.segments(X_df).items():
            m = mask.to_numpy()
            if m.sum() == 0:
                continue
            rows.append({"segment": label, "n": int(m.sum()),
                         "share": m.mean(),
                         "actual_buy_rate": y[m].mean(),
                         "model_mean_pred": proba[m].mean(),
                         "gap": proba[m].mean() - y[m].mean(),
                         "within_auc": roc_auc_score(y[m], proba[m])
                         if 0 < y[m].mean() < 1 else np.nan})
        return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)

    def plot(self, seg_table):
        t = seg_table.iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 6))
        yy = np.arange(len(t))
        ax.barh(yy + 0.2, t.actual_buy_rate, height=0.4, color="seagreen", label="actual")
        ax.barh(yy - 0.2, t.model_mean_pred, height=0.4, color="slateblue", label="model mean pred")
        ax.set_yticks(yy, t.segment, fontsize=8)
        ax.legend(); ax.grid(alpha=0.3, axis="x")
        ax.set_title("EDA facts vs model beliefs per segment")
        plt.tight_layout(); plt.show()
        return fig

    def rule_score(self, X_df):
        seg = self.segments(X_df)
        con = _coerce_num(X_df.get("Environmental_Concern_Level")).fillna(3)
        low_map = con.map({1: 0.006, 2: 0.024, 3: 0.120, 4: 0.268, 5: 0.545}).fillna(0.12)
        med_map = con.map({1: 0.001, 2: 0.004, 3: 0.025, 4: 0.059, 5: 0.174}).fillna(0.025)
        base = np.select(
            [seg["anxiety_high (0.14% buy)"].to_numpy(),
             seg["anxiety_medium"].to_numpy()],
            [0.0014, med_map.to_numpy()],
            default=low_map.to_numpy())
        base = np.where(seg["income_gt_169972 (100% buyers)"].to_numpy(), 0.999, base)
        base = np.where(seg["commute_ge_83 (0 buyers in EDA)"].to_numpy(), 0.0, base)
        base = np.where(seg["income_dead_band_31k_42k (0 buyers)"].to_numpy(), 0.0008, base)
        base = np.where(seg["no_subsidy_no_homechg (0.4%)"].to_numpy(),
                        np.minimum(base, 0.004), base)
        return pd.Series(base, index=X_df.index)
