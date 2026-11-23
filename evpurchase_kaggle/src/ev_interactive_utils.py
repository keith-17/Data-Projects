"""
ev_interactive_utils.py
Interactive analysis studio: widget-driven panels for feature comparison,
derived-vs-original importance exploration, and threshold/segment diagnostics.
All widget callbacks are bound methods so notebooks stay function-free.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

try:
    from IPython.display import clear_output, display
except ImportError:
    def clear_output(wait=False):
        pass

    def display(obj, *args, **kwargs):
        print(obj)

from ev_analysis_utils import (_coerce_num, _cramers_v, _information_value,
                               _mutual_information, DerivedVsOriginalAudit,
                               EDASegmentAuditor)


def _coerce_outcome(series):
    """Map a column to a binary 0/1 outcome for EDA metrics."""
    num = _coerce_num(series)
    if num.notna().mean() > 0.8 and num.nunique(dropna=True) > 2:
        med = num.median()
        return num.fillna(med).ge(med).astype(int)
    mapped = series.map({
        'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
        'True': 1, 'False': 0, 'true': 1, 'false': 0,
        1: 1, 0: 0, 1.0: 1, 0.0: 0,
    })
    if mapped.notna().mean() > 0.5:
        return mapped.fillna(0).astype(int)
    codes = series.astype(str).astype('category').cat.codes
    return (codes >= codes.median()).astype(int)


class InteractiveAnalysisStudio:
    def __init__(self, X_df, y, extractor=None, pipeline=None,
                 eval_df=None, eval_y=None, eval_proba=None, random_state=42,
                 source_df=None, default_target=None, target_options=None):
        self.random_state = random_state
        self.prepared_ = (extractor.transform_dataframe(X_df)
                          if extractor is not None else X_df.copy())
        self.source_df_ = (source_df.reindex(self.prepared_.index)
                           if source_df is not None else None)
        self.default_target_ = default_target
        if target_options is not None:
            self.target_options_ = list(target_options)
        elif self.source_df_ is not None:
            skip = {'id', 'ID', 'Id'}
            self.target_options_ = [c for c in self.source_df_.columns
                                    if c not in skip]
        else:
            self.target_options_ = []
        init_target = default_target
        if init_target is None and self.target_options_:
            init_target = self.target_options_[0]
        if init_target and self.source_df_ is not None and init_target in self.source_df_.columns:
            self.y_series = _coerce_outcome(self.source_df_[init_target])
        else:
            self.y_series = pd.Series(np.asarray(y, dtype=int),
                                      index=self.prepared_.index)
        self.pipeline = pipeline
        self.eval_y = None if eval_y is None else np.asarray(eval_y, dtype=int)
        self.eval_proba = (None if eval_proba is None
                           else np.asarray(eval_proba, dtype=float))
        self.feature_options_ = list(self.prepared_.columns)
        self.imp_tbl_ = None
        if pipeline is not None:
            try:
                self.imp_tbl_ = DerivedVsOriginalAudit().importance_table(pipeline)
            except Exception:
                self.imp_tbl_ = None
        self.segments_ = {}
        if eval_df is not None:
            try:
                aligned = eval_df.reset_index(drop=True)
                self.segments_ = {k: v.reset_index(drop=True)
                                  for k, v in EDASegmentAuditor().segments(aligned).items()}
            except Exception:
                self.segments_ = {}

    # ----------------------------------------------------------
    def make_controls(self):
        import ipywidgets as w
        opts = self.feature_options_
        default_a = 'Annual_Income_USD' if 'Annual_Income_USD' in opts else opts[0]
        default_b = 'Buy_Score' if 'Buy_Score' in opts else opts[min(1, len(opts) - 1)]
        groups = (['original'] + list(self.imp_tbl_['group'].unique())
                  if self.imp_tbl_ is not None else ['original'])
        groups = list(dict.fromkeys(groups))
        tgt_opts = self.target_options_ or [self.default_target_ or 'target']
        default_tgt = (self.default_target_ if self.default_target_ in tgt_opts
                       else tgt_opts[0])
        self.controls_ = {
            'feature_a': w.Dropdown(options=opts, value=default_a,
                                    description='Independent A:'),
            'feature_b': w.Dropdown(options=opts, value=default_b,
                                    description='Independent B:'),
            'dependent': w.Dropdown(options=tgt_opts, value=default_tgt,
                                    description='Dependent:'),
            'n_sample': w.IntSlider(min=2000, max=60000, step=2000, value=20000,
                                    description='Sample:'),
            'origin_filter': w.SelectMultiple(
                options=groups,
                value=tuple(g for g in ('original', 'screenshot', 'recipe') if g in groups),
                description='Groups:', rows=7),
            'top_n': w.IntSlider(min=5, max=30, step=1, value=15, description='Top-N:'),
            'threshold': w.FloatSlider(min=0.05, max=0.95, step=0.05, value=0.5,
                                       description='Threshold:'),
            'segment': w.Dropdown(options=['all'] + list(self.segments_.keys()),
                                  value='all', description='Segment:'),
        }
        return self.controls_

    def _target_series(self, dependent):
        if (self.source_df_ is not None and dependent in self.source_df_.columns):
            return _coerce_outcome(self.source_df_[dependent])
        return self.y_series

    # ----------------------------------------------------------
    def _feature_stats(self, s, y):
        num = _coerce_num(s)
        if s.dtype == object or num.notna().mean() < 0.5:
            v = s.astype(str)
            kind = 'cat'
            codes = v.astype('category').cat.codes
            a = roc_auc_score(y, codes) if codes.nunique() > 1 else 0.5
            a = max(a, 1 - a)
            cv = _cramers_v(v, y)
            mat = codes.to_numpy().reshape(-1, 1)
        else:
            v = num.fillna(num.median())
            kind = 'num'
            a = max(roc_auc_score(y, v), 1 - roc_auc_score(y, v))
            cv = _cramers_v(pd.qcut(v, q=10, duplicates='drop').astype(str), y)
            mat = v.to_numpy().reshape(-1, 1)
        return {'kind': kind, 'n_unique': int(s.nunique()), 'auc': a,
                'cramers_v': cv, 'iv': _information_value(s, y),
                'mi': _mutual_information(mat, y, self.random_state)}

    def _target_rate_plot(self, s, y, ax, title):
        num = _coerce_num(s)
        if s.dtype == object or num.notna().mean() < 0.5:
            g = (pd.DataFrame({'v': s.astype(str), 'y': y})
                 .groupby('v')['y'].agg(['mean', 'size'])
                 .sort_values('mean', ascending=False).head(10))
            ax.bar(g.index, g['mean'], color='teal')
            ax.tick_params(axis='x', rotation=45)
        else:
            v = num.fillna(num.median())
            b = pd.qcut(v, q=10, duplicates='drop')
            g = pd.DataFrame({'b': b.astype(str), 'y': y}).groupby('b', observed=True)['y'].mean()
            ax.bar(range(len(g)), g.values, color='mediumpurple')
            ax.set_xticks(range(len(g)),
                          [x.split(',')[-1].strip(']') for x in g.index],
                          rotation=45, fontsize=7)
        ax.set_ylabel('buy rate')
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='y')

    def _dist_plot(self, s, y, ax, title):
        num = _coerce_num(s)
        if s.dtype == object or num.notna().mean() < 0.5:
            pd.crosstab(s.astype(str), y).head(10).plot.bar(
                ax=ax, color=['tab:red', 'tab:green'])
            ax.tick_params(axis='x', rotation=45)
        else:
            v = num.fillna(num.median())
            ax.hist(v[y == 0], bins=40, alpha=0.6, color='tab:red', density=True, label='y=0')
            ax.hist(v[y == 1], bins=40, alpha=0.6, color='tab:green', density=True, label='y=1')
            ax.legend(fontsize=7)
        ax.set_title(title)

    # ----------------------------------------------------------
    def update_compare(self, feature_a, feature_b, dependent, n_sample):
        clear_output(wait=True)
        n = min(int(n_sample), len(self.prepared_))
        idx = self.prepared_.sample(n=n, random_state=self.random_state).index
        sa, sb = self.prepared_.loc[idx, feature_a], self.prepared_.loc[idx, feature_b]
        y = self._target_series(dependent).loc[idx].to_numpy()
        stats = pd.DataFrame({'A: ' + feature_a: self._feature_stats(sa, y),
                              'B: ' + feature_b: self._feature_stats(sb, y)}).T
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        self._target_rate_plot(sa, y, axes[0, 0], feature_a + ': buy rate')
        self._target_rate_plot(sb, y, axes[0, 1], feature_b + ': buy rate')
        self._dist_plot(sa, y, axes[1, 0], feature_a + ': distribution by class')
        self._dist_plot(sb, y, axes[1, 1], feature_b + ': distribution by class')
        fig.suptitle(f'Independent: {feature_a} vs {feature_b} | Dependent: {dependent}',
                     fontweight='bold')
        fig.tight_layout()
        plt.show()
        display(stats.style.background_gradient(
            cmap='magma', subset=['auc', 'cramers_v', 'iv', 'mi'])
            .format({'auc': '{:.4f}', 'cramers_v': '{:.4f}',
                     'iv': '{:.3f}', 'mi': '{:.4f}'}))

    # ----------------------------------------------------------
    def update_importance(self, origin_filter, top_n):
        clear_output(wait=True)
        if self.imp_tbl_ is None:
            print('⚠️ No fitted pipeline importances available.')
            return
        groups = list(origin_filter) if len(origin_filter) else list(self.imp_tbl_['group'].unique())
        t = self.imp_tbl_[self.imp_tbl_['group'].isin(groups)]
        share = self.imp_tbl_.groupby('group')['importance'].sum().sort_values(ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        top = t.sort_values('importance', ascending=False).head(int(top_n)).iloc[::-1]
        axes[0].barh(top['root'] + ' [' + top['group'] + ']', top['importance'],
                     color=['seagreen' if o == 'derived' else 'slategray' for o in top['origin']])
        axes[0].set_title('Top-' + str(int(top_n)) + ' importances in selected groups')
        share.plot.bar(ax=axes[1],
                       color=['teal' if g in groups else 'lightgray' for g in share.index])
        axes[1].set_title('Importance share by group (teal = selected)')
        for ax in axes:
            ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        sel_share = share.reindex(groups).sum() / max(share.sum(), 1e-12)
        derived_share = float(self.imp_tbl_.loc[self.imp_tbl_['origin'] == 'derived', 'importance'].sum())
        print(f'Selected groups hold {sel_share:.1%} of total importance | '
              f'overall derived share = {derived_share:.1%} | '
              f'overall original share = {1 - derived_share:.1%}')

    # ----------------------------------------------------------
    def update_diag(self, threshold, segment):
        clear_output(wait=True)
        if self.eval_proba is None or self.eval_y is None:
            print('⚠️ No holdout predictions available for this panel.')
            return
        mask = np.ones(len(self.eval_y), dtype=bool)
        if segment != 'all' and segment in self.segments_:
            mask = self.segments_[segment].to_numpy()[:len(self.eval_y)]
        y = self.eval_y[mask]
        p = self.eval_proba[mask]
        preds = (p >= float(threshold)).astype(int)
        base = (self.eval_proba >= 0.5).astype(int)[mask]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ConfusionMatrixDisplay.from_predictions(y, preds, cmap='Blues',
                                                  ax=axes[0], values_format='d')
        axes[0].set_title(f'Confusion @ thr={float(threshold):.2f} | {segment}')
        metrics = pd.DataFrame({
            'at_threshold': [precision_score(y, preds, zero_division=0),
                             recall_score(y, preds, zero_division=0),
                             f1_score(y, preds, zero_division=0),
                             (y == preds).mean()],
            'at_0.50': [precision_score(y, base, zero_division=0),
                        recall_score(y, base, zero_division=0),
                        f1_score(y, base, zero_division=0),
                        (y == base).mean()],
        }, index=['precision', 'recall', 'f1', 'accuracy'])
        metrics.plot.bar(ax=axes[1], color=['darkorange', 'gray'])
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Your threshold vs default 0.50')
        plt.tight_layout()
        plt.show()
        display(metrics.style.background_gradient(cmap='RdYlGn').format(
            {'at_threshold': '{:.3f}', 'at_0.50': '{:.3f}'}))
        print(f'segment n={int(mask.sum()):,} | actual buy rate={y.mean():.2%} | '
              f'mean predicted={p.mean():.2%}')