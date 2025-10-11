import numpy as np
import pandas as pd

def spread_proxy(high, low):
    rng = (high - low).clip(lower=0)
    mid = 0.5 * (high + low)
    sp = (rng / mid).fillna(0.0)
    return 0.25 * sp

def realized_vol(close, win=60):
    r = close.pct_change()
    return r.rolling(win).std().fillna(0.0)

def participation(dollar_trade, price, volume):
    dollar_bar = (price * volume).replace(0, np.nan)
    return (dollar_trade / dollar_bar).fillna(0.0).clip(lower=0.0)

def exec_price(side, mid, half_spread, sigma, dt_min, part, k=0.6, lam=2e-4):
    drift = k * sigma * np.sqrt(max(dt_min, 1.0))
    impact = lam * (part ** 2)
    if side > 0:
        px = mid * (1.0 + half_spread + drift + impact)
    elif side < 0:
        px = mid * (1.0 - half_spread - drift - impact)
    else:
        px = mid
    return px

def turnover_l1(w_new, w_old):
    """L1 turnover penalty: sum of absolute changes in portfolio weights."""
    return np.abs(w_new - w_old).sum()
