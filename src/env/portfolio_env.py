import numpy as np
import pandas as pd
from src.utils.costs import spread_proxy, realized_vol, participation, exec_price, turnover_l1

class PortfolioEnv:
    def __init__(
        self,
        prices,
        freq_min=1,
        start_equity=1_000_000,
        part_cap=0.05,
        k=0.6,
        lam=2e-4,
        gamma_bar=11.0,
        eta_turnover=0.001
    ):
        self.prices = prices
        self.close = prices.loc[:, pd.IndexSlice[:, "Close"]]
        self.high = prices.loc[:, pd.IndexSlice[:, "High"]]
        self.low = prices.loc[:, pd.IndexSlice[:, "Low"]]
        self.vol = prices.loc[:, pd.IndexSlice[:, "Volume"]]
        self.mid = self.close

        # Cost and risk model components
        self.half_spread = spread_proxy(self.high, self.low)
        self.sigma = realized_vol(self.close, win=60)

        self.part_cap = part_cap
        self.k = k
        self.lam = lam
        self.freq_min = freq_min
        self.gamma = gamma_bar / (252 * 390)  # annualize risk cost
        self.eta_turnover = eta_turnover
        self.start_equity = start_equity
        self.tickers = self.close.columns.get_level_values(0).unique()

        self.reset()

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------
    def reset(self):
        self.t = 1
        self.equity = float(self.start_equity)
        self.w = np.zeros(len(self.tickers))
        self.cash = self.equity
        return self._obs()

    # ------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------
    def _obs(self):
        return {"weights": self.w.copy(), "equity": float(self.equity)}

    # ------------------------------------------------------
    # STEP
    # ------------------------------------------------------
    def step(self, w_target):
        # --- Safety clamp ---
        w_target = np.array(w_target).flatten()
        w_target = np.clip(w_target, 0, 1)
        if w_target.sum() > 0:
            w_target /= w_target.sum()

        # --- Time + current prices ---
        idx = self.close.index[self.t]
        idx_next = self.close.index[min(self.t + 1, len(self.close) - 1)]

        mid = np.nan_to_num(self.mid.loc[idx].values, nan=0.0, posinf=0.0, neginf=0.0)
        hs = np.nan_to_num(self.half_spread.loc[idx].values, nan=0.0, posinf=0.0, neginf=0.0)
        sg = np.nan_to_num(self.sigma.loc[idx].values, nan=1e-6, posinf=1e-6, neginf=1e-6)
        vol = np.nan_to_num(self.vol.loc[idx].values, nan=1.0, posinf=1.0, neginf=1.0)

        # --- Compute trade ---
        delta_w = w_target - self.w
        dollar_trade = np.abs(delta_w) * self.equity
        part = participation(pd.Series(dollar_trade), pd.Series(mid), pd.Series(vol)).values
        part = np.nan_to_num(np.clip(part, 0, self.part_cap), nan=0.0)
        side = np.sign(delta_w)

        exec_px = np.array([
            exec_price(s, m, h, sgm, self.freq_min, p, self.k, self.lam)
            for s, m, h, sgm, p in zip(side, mid, hs, sg, part)
        ])
        exec_px = np.nan_to_num(exec_px, nan=mid, posinf=mid, neginf=mid)

        # --- Trade cash impact ---
        shares = np.nan_to_num((np.abs(delta_w) * self.equity) / np.maximum(mid, 1e-12))
        trade_cash = np.sum(shares * exec_px * side)

        # --- Update positions ---
        dollar_pos = self.w * self.equity
        dollar_pos += shares * exec_px * side
        self.equity = np.maximum(self.equity - np.abs(trade_cash), 1.0)
        self.cash = max(self.start_equity - np.sum(dollar_pos), 0.0)

        next_px = np.nan_to_num(self.mid.loc[idx_next].values, nan=mid, posinf=mid, neginf=mid)
        dollar_pos = dollar_pos * (next_px / np.maximum(mid, 1e-12))
        self.equity = float(np.nan_to_num(np.sum(dollar_pos) + self.cash, nan=self.start_equity))
        self.w = np.clip(np.nan_to_num(dollar_pos / np.maximum(self.equity, 1e-12)), 0, 1)

        # --- Penalties ---
        var_diag = np.nan_to_num((self.sigma.loc[idx] ** 2).values, nan=0.0)
        risk_pen = 0.5 * self.gamma * float(np.dot(self.w, var_diag * self.w))
        tvr = turnover_l1(self.w, self.w - delta_w)
        tvr_pen = self.eta_turnover * tvr

        # --- Reward ---
        prev_idx = self.close.index[self.t - 1]
        r_bar = np.nan_to_num(
            self.mid.loc[idx_next].values / np.maximum(self.mid.loc[prev_idx].values, 1e-12) - 1.0,
            nan=0.0,
        )
        pnl_ret = float(np.dot(self.w, r_bar))

        drift_pen = 0.001 * np.square(delta_w).sum()

        reward = (pnl_ret - risk_pen - tvr_pen - drift_pen) / (np.abs(risk_pen) + 1e-6)

        # --- Stability guards ---
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self.equity = float(np.nan_to_num(self.equity, nan=self.start_equity, posinf=self.start_equity, neginf=self.start_equity))

        # Scale reward
        reward = float(np.clip(np.tanh(reward), -1, 1))

        info = {
            "pnl_ret": pnl_ret,
            "risk_pen": risk_pen,
            "tvr_pen": tvr_pen,
            "equity": float(self.equity)
        }

        self.t += 1
        done = self.t >= len(self.close) - 1
        return self._obs(), reward, done, info
