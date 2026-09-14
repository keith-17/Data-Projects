"""
ev_data_utils.py

Data loading utilities for the Kaggle EV purchase prediction competition.

This module is inspired by the dynamic path-routing style used in the
multibranch notebook and data_utils.py, but adapted for this tabular EV task.

It supports:
- local data folders, e.g. ./data/train.csv
- notebook folders, e.g. project/notebooks with data one level up
- Kaggle input folders, e.g. /kaggle/input/<dataset>/train.csv
- fallback to sample_ev.csv for fast local smoke tests
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd


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


def _unique_paths(paths):
    """
    Remove duplicate Path objects while preserving order.
    """
    unique = []
    for p in paths:
        if p not in unique:
            unique.append(p)
    return unique


def find_data_root(
    local_dir: str = "data",
    kaggle_root: str = "/kaggle/input",
    required_files=("train.csv",),
    verbose: bool = True,
) -> Path:
    """
    Robustly find the data root directory.

    Search order:
    1. Exact provided path.
    2. Upwards from current working directory.
    3. Upwards from this file's directory.
    4. Upwards from calling script/notebook directory.
    5. Kaggle input folder.
    """
    if isinstance(required_files, str):
        required_files = (required_files,)

    required_files = tuple(Path(f) for f in required_files)

    def check_dir(d: Path):
        target = d / local_dir
        if target.exists() and target.is_dir():
            if all((target / f).exists() for f in required_files):
                return target
        return None

    # 1. Exact path provided.
    exact_path = Path(local_dir)
    if exact_path.exists() and exact_path.is_dir():
        if all((exact_path / f).exists() for f in required_files):
            if verbose:
                print(f"✅ Using exact path: {exact_path.resolve()}")
            return exact_path

    # Gather starting points.
    search_starts = []

    # Current working directory.
    search_starts.append(Path.cwd())

    # Directory of this file.
    try:
        search_starts.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    # Directory of calling script/notebook.
    try:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_globals.get("__file__")
            if fname:
                search_starts.append(Path(fname).resolve().parent)
            frame = frame.f_back
    except Exception:
        pass

    unique_starts = _unique_paths(search_starts)

    # 2-4. Search upwards.
    for start_path in unique_starts:
        current = start_path.resolve()
        for parent in [current] + list(current.parents):
            res = check_dir(parent)
            if res:
                if verbose:
                    print(f"✅ Found data folder: {res}")
                return res

    # 5. Kaggle environment.
    kaggle_path = Path(kaggle_root)
    if kaggle_path.exists():
        first_required = required_files[0].name

        # Candidate parents containing the first required file.
        candidate_parents = [
            candidate.parent
            for candidate in kaggle_path.rglob(first_required)
        ]

        # Prefer parents where all required files exist.
        for parent in candidate_parents:
            if all((parent / f).exists() for f in required_files):
                if verbose:
                    print(f"✅ Using Kaggle data folder: {parent}")
                return parent

    raise FileNotFoundError(
        f"❌ Could not find dataset.\n"
        f"Searched upwards from: {[str(p) for p in unique_starts]}\n"
        f"Looked for a folder named '{local_dir}' containing {required_files}.\n"
        f"Current directory: {Path.cwd()}\n"
        f"Kaggle root checked: {kaggle_root}\n"
        "Please ensure train.csv is in a local 'data/' folder or attached Kaggle dataset."
    )


def _find_sample_path(
    sample_name: str = "sample_ev.csv",
    local_dir: str = "data",
    kaggle_root: str = "/kaggle/input",
) -> Path:
    """
    Find sample_ev.csv locally or on Kaggle.
    """
    direct_candidates = [
        Path(sample_name),
        Path(local_dir) / sample_name,
        Path.cwd() / sample_name,
        Path.cwd() / local_dir / sample_name,
    ]

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    # Local recursive search.
    for candidate in Path.cwd().rglob(sample_name):
        return candidate

    # Kaggle recursive search.
    kaggle_path = Path(kaggle_root)
    if kaggle_path.exists():
        for candidate in kaggle_path.rglob(sample_name):
            return candidate

    return None


def load_ev_data(
    local_dir: str = "data",
    train_name: str = "train.csv",
    test_name: str = "test.csv",
    sample_name: str = "sample_ev.csv",
    target_col: str = "Will_Buy_EV",
    kaggle_root: str = "/kaggle/input",
    random_state: int = 42,
):
    """
    Load train and test data for the EV competition.

    Returns:
        train_df: DataFrame with target column.
        test_df: DataFrame without required target, or None.
        source: Description of where data came from.
    """
    try:
        data_root = find_data_root(
            local_dir=local_dir,
            kaggle_root=kaggle_root,
            required_files=(train_name,),
            verbose=True,
        )

        train_path = data_root / train_name
        train_df = pd.read_csv(train_path)

        test_path = data_root / test_name
        if test_path.exists():
            test_df = pd.read_csv(test_path)
        else:
            test_df = None

        source = f"Train file: {train_path}"

    except FileNotFoundError:
        sample_path = _find_sample_path(
            sample_name=sample_name,
            local_dir=local_dir,
            kaggle_root=kaggle_root,
        )

        if sample_path is None:
            raise FileNotFoundError(
                "❌ Could not find train.csv or sample_ev.csv.\n"
                "Please place train.csv in ./data/ or attach the Kaggle dataset.\n"
                "For local smoke tests, place sample_ev.csv in ./data/."
            )

        full_df = pd.read_csv(sample_path)

        # In sample mode, use the sample as train and also create a submission-like test frame.
        train_df = full_df.copy()

        if target_col in full_df.columns:
            test_df = full_df.drop(columns=[target_col], errors="ignore").copy()
        else:
            test_df = full_df.copy()

        source = f"SAMPLE MODE: {sample_path}"

    # Clean target in train.
    if target_col in train_df.columns:
        train_df[target_col] = train_df[target_col].map(_coerce_binary)
        train_df = train_df.dropna(subset=[target_col]).copy()
        train_df[target_col] = train_df[target_col].astype(int)

    # If test accidentally has target, clean but do not force removal.
    if test_df is not None and target_col in test_df.columns:
        test_df[target_col] = test_df[target_col].map(_coerce_binary)

    return train_df, test_df, source