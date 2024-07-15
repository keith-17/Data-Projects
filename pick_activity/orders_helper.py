import pandas as pd
import numpy as np


def clean_by_dropping_records(df: pd.DataFrame, pick_type_list: list = ['GNR']) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['ORDER_NUMBER', 'ORDERED_PRODUCT_ID'])
    df = df[~df['PICK_TYPE'].isin(pick_type_list)]
    return df


def clean_by_unit_measure(df: pd.DataFrame) -> pd.DataFrame:
    fulfilled_gram_rows = (df['PICKED_UNIT_OF_MEASURE'] == 'GRAM') & (df['PICK_TYPE'] == 'NORMAL')
    df.loc[fulfilled_gram_rows, 'FINAL_QTY'] = df.loc[fulfilled_gram_rows, 'ORDERED_QTY']

    remaining_rows = df['PICKED_UNIT_OF_MEASURE'] == 'EACH'
    df.loc[remaining_rows, 'FINAL_QTY'] = df.loc[remaining_rows, 'QTY']
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['EVENT_TIME'] = pd.to_datetime(df['EVENT_TIME'], format='%H:%M:%S')
    df['event_hour'] = df['EVENT_TIME'].dt.hour
    df['FINAL_QTY'] = np.nan
    return df


def clean_by_pick_type(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['PICK_TYPE'] == 'NAG', 'FINAL_QTY'] = 0
    return df


def clean_order_data(df: pd.DataFrame) -> pd.DataFrame:
    order_df = clean_by_dropping_records(df)
    order_df = clean_columns(order_df)
    order_df = clean_by_pick_type(order_df)
    order_df = clean_by_unit_measure(order_df)
    return order_df


def create_delivery_df(df: pd.DataFrame) ->pd.DataFrame:
    deliveries_df = df.groupby(['ORDER_NUMBER']).agg(**{
        'max_time': ('EVENT_TIME', 'max'),
        'min_time': ('EVENT_TIME', 'min'),
        'operator_count': ('PICKER_ID', 'nunique'),
        'product_count': ('PICKED_PRODUCT_ID', 'nunique'),
        'units_count': ('FINAL_QTY', 'sum')
    })

    deliveries_df['packing_time_time_delta'] = deliveries_df['max_time'] - deliveries_df['min_time']
    deliveries_df['packing_time'] = deliveries_df['packing_time_time_delta'].dt.total_seconds() / 3600
    deliveries_df = deliveries_df.sort_values(by='packing_time').reset_index()
    return deliveries_df


def summarise_delivery_pack_time(df: pd.DataFrame) -> pd.DataFrame:
    bin_counts = [0, 1, 2, 3, 4, 5, 6, 7, float('inf')]
    labels = ['< 1 hour', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours', '5-6 hours', '6-7 hours', '> 7 hours']
    df['Banded Pack Time'] = pd.cut(df['packing_time'], bins=bin_counts, labels=labels, include_lowest=True)
    return df


def summarise_delivery_metrics(df:pd.DataFrame) -> pd.DataFrame:
    df['Picking Speed'] = df['units_count'] / df['packing_time']
    df['Picking Speed'] = df['Picking Speed'].replace(float('inf'), 0)
    return df


def get_delivery_data(df: pd.DataFrame) -> pd.DataFrame:
    deliveries_df = create_delivery_df(df)
    deliveries_df = summarise_delivery_pack_time(deliveries_df)
    deliveries_df = summarise_delivery_metrics(deliveries_df)
    return deliveries_df


def summarise_delivery_data(df: pd.DataFrame) -> pd.DataFrame:
    deliveries_summary_df = df.groupby(['Banded Pack Time']).agg(**{
        'Delivery Count': ('ORDER_NUMBER', 'nunique')})

    deliveries_summary_df['Proportion Deliveries'] = (deliveries_summary_df['Delivery Count'] / deliveries_summary_df[
        'Delivery Count'].sum()) * 100

    deliveries_summary_df['Proportion Deliveries'] = round(deliveries_summary_df['Proportion Deliveries'], 2)

    deliveries_summary_df['Deliveries Cumsum'] = deliveries_summary_df['Proportion Deliveries'].cumsum()

    return deliveries_summary_df


def band_operator_labels(df: pd.DataFrame) -> pd.DataFrame:
    operator_bins = np.arange(0, 28, 4)
    operator_labels = ['0 - 4', '4 - 8', '8 - 12', '12 - 16', '16 - 20', '20 - 24']
    df['Average Orders Per Hour Bin'] = pd.cut(df['Average Orders Per Hour'],
                                               bins=operator_bins, labels=operator_labels)
    return df


def aggregate_operator_line_data(df: pd.DataFrame) -> pd.DataFrame:
    operator_pack_df = df.groupby('PICKER_ID').agg(**{
        'Average Orders Per Hour': ('Total Orders', 'mean')})

    operator_pack_df = operator_pack_df.sort_values(by='Average Orders Per Hour').reset_index().reset_index().rename(
        columns={'index': 'count'})
    operator_pack_df['Percentage Cumsum'] = operator_pack_df['count'].cumsum() / operator_pack_df['count'].sum() * 100

    operator_pack_df = band_operator_labels(operator_pack_df)

    return operator_pack_df
