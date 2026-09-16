"""
ev_cluster_utils.py
Interactive unsupervised clustering explorer (widget callbacks live here,
so notebooks never define functions).
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
    _UMAP = True
except Exception:
    _UMAP = False

try:
    from hdbscan import HDBSCAN
    _HDBSCAN = True
except Exception:
    _HDBSCAN = False


class ClusterExplorer:
    """Cluster engineered features on selected columns and visualise."""

    def __init__(self, X_df, y=None, extractor=None, random_state=42):
        self.random_state = random_state
        self.y = y
        if extractor is not None:
            self.prepared_ = extractor.transform_dataframe(X_df)
        else:
            self.prepared_ = X_df.copy()
        self.default_features_ = [
            c for c in ["Annual_Income_USD", "Age", "Daily_Commute_km",
                        "Range_Anxiety_Level", "Environmental_Concern_Level",
                        "Subsidy_Available", "Buy_Score", "Charging_Density"]
            if c in self.prepared_.columns
        ] or list(self.prepared_.columns[:6])

    # ----------------------------------------------------------
    def make_controls(self):
        import ipywidgets as w
        methods = ["kmeans", "gmm", "agglomerative", "dbscan"]
        if _HDBSCAN:
            methods.append("hdbscan")
        reductions = ["pca", "tsne"]
        if _UMAP:
            reductions.append("umap")
        color_opts = ["cluster_label"] + list(self.prepared_.columns)
        if self.y is not None:
            color_opts = color_opts + ["__target__"]
        self.controls_ = {
            "method": w.Dropdown(options=methods, value=methods[0],
                                 description="Cluster:"),
            "reduction": w.Dropdown(options=reductions, value="pca",
                                    description="Reduce:"),
            # FIX: min/max/value must ALL be keyword arguments
            "n_clusters": w.IntSlider(min=2, max=10, step=1, value=4,
                                      description="k:"),
            "features": w.SelectMultiple(
                options=list(self.prepared_.columns),
                value=tuple(self.default_features_),
                description="Features:", rows=8),
            "scale": w.Checkbox(value=True, description="Standardize"),
            "color_by": w.Dropdown(options=color_opts, value="cluster_label",
                                   description="Color:"),
            # FIX: same keyword-argument fix here
            "sample_size": w.IntSlider(min=2000, max=40000, step=2000,
                                       value=15000, description="Sample:"),
        }
        return self.controls_

    # ----------------------------------------------------------
    def update(self, method, reduction, n_clusters, features, scale,
               color_by, sample_size):
        from IPython.display import clear_output
        clear_output(wait=True)
        feats = list(features) if len(features) else self.default_features_
        df = self.prepared_.loc[:, feats].copy()
        if any(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
            df = pd.get_dummies(df)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        n = min(int(sample_size), len(df))
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(df), size=n, replace=False)
        M = df.iloc[idx].to_numpy(dtype=float)
        if scale:
            M = StandardScaler().fit_transform(M)

        # 2-D embedding (clustering + plotting happen in this space)
        if reduction == "tsne":
            Z = TSNE(n_components=2, init="pca", learning_rate="auto",
                     random_state=self.random_state).fit_transform(M)
        elif reduction == "umap" and _UMAP:
            Z = UMAP(n_components=2,
                     random_state=self.random_state).fit_transform(M)
        else:
            Z = PCA(n_components=2,
                    random_state=self.random_state).fit_transform(M)

        # clustering
        if method == "kmeans":
            labels = KMeans(n_clusters=n_clusters, n_init=10,
                            random_state=self.random_state).fit_predict(Z)
        elif method == "gmm":
            labels = GaussianMixture(n_components=n_clusters, n_init=3,
                                     random_state=self.random_state
                                     ).fit_predict(Z)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(
                n_clusters=n_clusters).fit_predict(Z)
        elif method == "dbscan":
            labels = DBSCAN(eps=0.5, min_samples=15).fit_predict(Z)
        else:
            labels = HDBSCAN(min_cluster_size=50).fit_predict(Z)

        n_lab = len(set(labels)) - (1 if -1 in labels else 0)
        sil = ""
        if n_lab >= 2 and n_lab < len(Z):
            sil = f" | silhouette={silhouette_score(Z, labels):.3f}"
        sizes = pd.Series(labels).value_counts().sort_index()

        # plot
        fig, ax = plt.subplots(figsize=(9, 6.5))
        if color_by == "cluster_label":
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10",
                            s=8, alpha=0.75)
            fig.colorbar(sc, ax=ax, label="cluster")
        else:
            vals = (self.y.iloc[idx].to_numpy(dtype=float)
                    if color_by == "__target__"
                    else pd.to_numeric(self.prepared_[color_by].iloc[idx],
                                       errors="coerce").fillna(0).to_numpy())
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=vals, cmap="viridis",
                            s=8, alpha=0.75)
            fig.colorbar(sc, ax=ax, label=color_by)
        ax.set_title(f"{method.upper()} on {reduction.upper()} "
                     f"({len(feats)} features, n={n:,}){sil}")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        plt.tight_layout()
        plt.show()

        print(f"clusters found: {n_lab}{sil}")
        print(sizes.to_string())
        if self.y is not None and color_by == "cluster_label":
            rate = pd.Series(self.y.iloc[idx].to_numpy(dtype=float)
                             ).groupby(labels).mean()
            print("target (buy) rate per cluster:")
            print(rate.round(4).to_string())