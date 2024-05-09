"""
From ChatGPT:
If the mark price is higher than the index price, arbitrageurs might sell the overvalued derivative contract and buy the
underlying asset simultaneously, aiming to profit from the convergence of prices. Conversely, if the mark price is lower
than the index price, arbitrageurs might buy the undervalued derivative contract and sell the underlying asset.

GATEIO Futures: has futures, also has regular (non-fractional?)
OKX: has swap contracts?
Binance: regular, fractional
ByBit: fractional, less than 1
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gzip

filepath = "04/20240401_book_updates.csv.gz"

df = pd.read_csv(filepath, compression = "gzip")

print(df)


