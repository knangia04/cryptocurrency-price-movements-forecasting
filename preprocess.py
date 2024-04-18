import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cal_mid_price(df):
    return (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2

def cal_spread(df):
    return df['ASK_PRICE_1'] - df['BID_PRICE_1']

def process_data(df):
    df['MID_PRICE'] = cal_mid_price(df)
    df['SPREAD'] = cal_spread(df)
    df["COLLECTION_TIME"] = pd.to_datetime(df["COLLECTION_TIME"]).values.astype(np.int64)
    df['Time_Delta'] = df['COLLECTION_TIME'].diff()
    df.drop(["COLLECTION_TIME", "MESSAGE_ID", "MESSAGE_TYPE", "SYMBOL"], axis=1, inplace=True)
    df["COLLECTION_TIME"] = pd.to_datetime(df["COLLECTION_TIME"]).values.astype(np.int64)
    columns_to_scale = df.columns.difference(['Time_Delta'])
    # Scale the selected columns
    minmax_scaler = MinMaxScaler()
    df[columns_to_scale] = minmax_scaler.fit_transform(df[columns_to_scale])
    return df

