import pandas as pd
import numpy as np

def cal_mid_price(df):
    return (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2

def cal_spread(df):
    return df['ASK_PRICE_1'] - df['BID_PRICE_1']

def process_data(df):
    df['MID_PRICE'] = cal_mid_price(df)
    df['SPREAD'] = cal_spread(df)
    df.drop(["COLLECTION_TIME", "MESSAGE_ID", "MESSAGE_TYPE", "SYMBOL"], axis=1, inplace=True)
    return df

# Example usage
df = pd.DataFrame({'BID_PRICE_1': [10, 20, 30], 'ASK_PRICE_1': [15, 25, 35]})
processed_df = process_data(df)
print(processed_df)