import numpy as np
import pandas as pd
from src.utils.costs import spread_proxy, realized_vol, participation, exec_price, turnover_l1


class PortfolioEnv:
    """
    A simulation environment for multi-asset portfolio trading with market frictions,
    transaction costs, and risk penalties.

    Attributes
    ----------
    prices : DataFrame
        Multi-indexed price panel containing OHLCV data for multiple tickers.
    freq_min : int
        Time step size in minutes.
    start_equity : float
        Initial portfolio equity.
    part_cap : float
        Maximum participation rate per trade.
    k : float
        Drift scaling coefficient in execution model.
    lam : float
        Market impact parameter.
    gamma_bar : float
        Annualized risk aversion, converted to per-minute scale internally.
    eta_turnover : float
        Turnover penalty weight.
    """

    def __init__(
        self,
        prices,
        freq_min=1,
        start_equity=1_000_000,
        part_cap=0.05,
        k=0.6,
        lam=2e-4,
        gamma_bar=12,
        eta_turnover=0.001
    ):
        """
        Initializes the environment with price data and trading parameters.
        Precomputes spread and volatility proxies used in transaction and risk modeling.
        """
        self.prices = prices
        self.close = prices.loc[:, pd.IndexSlice[:, "Close"]]
        self.high = prices.loc[:, pd.IndexSlice[:, "High"]]
        self.low = prices.loc[:, pd.IndexSlice[:, "Low"]]
        self.vol = prices.loc[:, pd.IndexSlice[:, "Volume"]]
        self.mid = self.close

        self.half_spread = spread_proxy(self.high, self.low)
        self.sigma = realized_vol(self.close, win=60)

        self.part_cap = part_cap
        self.k = k
        self.lam = lam
        self.freq_min = freq_min
        self.gamma = gamma_bar / (252 * 390)
        self.eta_turnover = eta_turnover
        self.start_equity = start_equity
        self.tickers = self.close.columns.get_level_values(0).unique()

        self.reset()

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns
        -------
        dict
            Initial observation containing portfolio weights and equity.
        """
        self.t = 1
        self.equity = float(self.start_equity)
        self.w = np.zeros(len(self.tickers))
        self.q = np.zeros(len(self.tickers))
        self.cash = float(self.start_equity)
        return self._obs()

    def _obs(self):
        """
        Returns the current observable state of the environment.

        Returns
        -------
        dict
            Dictionary with current weights and equity.
        """
        return {"weights": self.w.copy(), "equity": float(self.equity)}

    def step(self, w_target):
        """
        Advances the environment by one step given a target weight allocation.

        Parameters
        ----------
        w_target : array-like
            Target portfolio weights for all tickers.

        Returns
        -------
        tuple
            observation : dict
                Updated weights and equity.
            reward : float
                Scalar reward value after applying penalties.
            done : bool
                Whether the simulation has reached the final time step.
            info : dict
                Diagnostic metrics including pnl, penalties, and equity.
        """
        w_target = np.array(w_target).flatten()
        w_target = np.clip(w_target, 0, 1)
        if w_target.sum() > 0:
            w_target /= w_target.sum()

        idx = self.close.index[self.t]
        idx_next = self.close.index[min(self.t + 1, len(self.close) - 1)]

        mid = np.nan_to_num(self.mid.loc[idx].values, nan=0.0, posinf=0.0, neginf=0.0)
        hs = np.nan_to_num(self.half_spread.loc[idx].values, nan=0.0, posinf=0.0, neginf=0.0)
        sg = np.nan_to_num(self.sigma.loc[idx].values, nan=1e-6, posinf=1e-6, neginf=1e-6)
        vol = np.nan_to_num(self.vol.loc[idx].values, nan=1.0, posinf=1.0, neginf=1.0)

        dollar_pos_now = self.q * mid
        port_now = float(self.cash + np.sum(dollar_pos_now))
        self.equity = max(port_now, 1.0)

        delta_w = w_target - self.w

        desired_notional = delta_w * self.equity
        part = participation(pd.Series(np.abs(desired_notional)), pd.Series(mid), pd.Series(vol)).values
        part = np.nan_to_num(np.clip(part, 0, self.part_cap), nan=0.0)
        feasible_notional = desired_notional * part
        signed_shares = np.nan_to_num(feasible_notional / np.maximum(mid, 1e-12))

        exec_px = np.array([
            exec_price(np.sign(dw), m, h, sgm, self.freq_min, p, self.k, self.lam)
            for dw, m, h, sgm, p in zip(delta_w, mid, hs, sg, part)
        ])
        exec_px = np.nan_to_num(exec_px, nan=mid, posinf=mid, neginf=mid)

        trade_cash_flow = float(np.sum(signed_shares * exec_px))
        self.cash -= trade_cash_flow
        self.q += signed_shares

        next_px = np.nan_to_num(self.mid.loc[idx_next].values, nan=mid, posinf=mid, neginf=mid)
        portfolio_value = float(self.cash + np.sum(self.q * next_px))
        self.equity = max(portfolio_value, 1.0)

        dollar_pos_next = self.q * next_px
        self.w = np.clip(np.nan_to_num(dollar_pos_next / np.maximum(self.equity, 1e-12)), 0, 1)

        var_diag = np.nan_to_num((self.sigma.loc[idx] ** 2).values, nan=0.0)
        risk_pen = 0.5 * self.gamma * float(np.dot(self.w, var_diag * self.w))

        prev_w = np.nan_to_num(dollar_pos_now / np.maximum(port_now, 1e-12))
        tvr = turnover_l1(self.w, prev_w)
        tvr_pen = self.eta_turnover * tvr

        prev_idx = self.close.index[self.t - 1]
        r_bar = np.nan_to_num(
            self.mid.loc[idx_next].values / np.maximum(self.mid.loc[prev_idx].values, 1e-12) - 1.0,
            nan=0.0,
        )
        pnl_ret = float(np.dot(self.w, r_bar))

        drift_pen = np.clip(1e-5 * np.square(delta_w).sum(), 0, 1e-3)

        raw = pnl_ret - risk_pen - tvr_pen - drift_pen
        reward = float(np.clip(np.tanh(raw * 20.0), -1.0, 1.0))

        info = {
            "pnl_ret": pnl_ret,
            "risk_pen": risk_pen,
            "tvr_pen": tvr_pen,
            "equity": float(self.equity)
        }

        self.t += 1
        done = self.t >= len(self.close) - 1

        return self._obs(), reward, done, info
