import pandas as pd
import numpy as np


def expand_booking_dates_fast(df):
    """Ultra-fast date expansion using numpy"""
    start_dates = df["started_at_dt"].dt.normalize()
    end_dates = df["closed_at_dt"].dt.normalize()
    days_diff = (end_dates - start_dates).dt.days + 1

    # Create arrays for expansion
    booking_indices = np.repeat(df.index.values, days_diff.values)
    day_offsets = np.concatenate([np.arange(days) for days in days_diff.values])

    # Create expanded dataframe
    expanded_df = df.iloc[booking_indices].copy()
    expanded_df['date_occupied'] = (
            start_dates.iloc[booking_indices].reset_index(drop=True) +
            pd.to_timedelta(day_offsets, unit='D')
    )

    return expanded_df.reset_index(drop=True)


def expand_booking_dates_optimized(df):
    """Fastest approach using pure pandas vectorization"""
    # Calculate date ranges directly
    starts = df['started_at_dt'].dt.normalize()
    ends = df['closed_at_dt'].dt.normalize()

    # Use list comprehension with zip (much faster than apply)
    df = df.copy()
    df['occupancy_days'] = [
        pd.date_range(s, e, freq='D')
        for s, e in zip(starts, ends)
    ]

    return df.explode('occupancy_days').rename(
        columns={'occupancy_days': 'date_occupied'}
    ).reset_index(drop=True)