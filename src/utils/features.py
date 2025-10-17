
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std()
    df["momentum_5"] = df["Close"].pct_change(5)
    df["momentum_20"] = df["Close"].pct_change(20)

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # Z-score
    df["zscore"] = (df["Close"] - df["Close"].rolling(20).mean()) / (df["Close"].rolling(20).std() + 1e-9)

    # Volume dynamics
    df["vol_change"] = df["Volume"].pct_change()

    return df.dropna()
def build_all_features(panel):
    tickers = panel.columns.get_level_values(0).unique()
    feat_list = []
    for t in tickers:
        df_t = panel[t].copy()
        df_feat = add_features(df_t)
        feat_list.append(df_feat)
    feat = pd.concat(feat_list, axis=1, keys=tickers)
    return feat
