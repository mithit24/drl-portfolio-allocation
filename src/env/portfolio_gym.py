import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.env.portfolio_env import PortfolioEnv


class PortfolioGym(gym.Env):
    """
    Gymnasium-compatible wrapper around the PortfolioEnv environment.

    This class provides a standardized reinforcement learning interface for
    portfolio management tasks, enabling interaction through observations,
    actions, and rewards in compliance with Gymnasium API conventions.

    Attributes
    ----------
    core : PortfolioEnv
        Core environment handling portfolio mechanics, execution, and reward logic.
    feature_df : DataFrame or None
        Optional feature matrix aligned with the time index of the price data.
    tickers : list
        List of asset tickers included in the environment.
    n_assets : int
        Number of tradable assets.
    observation_space : gym.spaces.Box
        Continuous space containing equity, portfolio weights, and optional features.
    action_space : gym.spaces.Box
        Continuous space representing target portfolio weights.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, feature_df=None, start_equity=1_000_000):
        """
        Initialize the Gym-compatible portfolio environment.

        Parameters
        ----------
        prices : DataFrame
            Multi-indexed price panel containing OHLCV data.
        feature_df : DataFrame, optional
            External feature matrix to augment observations.
        start_equity : float, default=1_000_000
            Initial portfolio value.
        """
        super().__init__()
        self.core = PortfolioEnv(prices, start_equity=start_equity)
        self.feature_df = feature_df
        self.tickers = self.core.tickers
        self.n_assets = len(self.tickers)

        feat_dim = 0 if feature_df is None else feature_df.shape[1]
        obs_dim = 1 + self.n_assets + feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        tuple
            observation : ndarray
                Normalized observation vector after reset.
            info : dict
                Auxiliary information (empty by default).
        """
        super().reset(seed=seed)
        obs_core = self.core.reset()
        obs = self._build_obs(obs_core)
        return obs.astype(np.float32), {}

    def _build_obs(self, core_obs):
        """
        Construct the observation vector from core environment data.

        Parameters
        ----------
        core_obs : dict
            Core observation dictionary containing weights and equity.

        Returns
        -------
        ndarray
            Flattened and normalized observation vector.
        """
        equity = float(core_obs["equity"])
        w = np.array(core_obs["weights"], dtype=np.float32)

        if self.feature_df is not None:
            t = min(self.core.t, len(self.feature_df) - 1)
            f = self.feature_df.iloc[t].to_numpy(dtype=np.float32)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            f = (f - np.mean(f)) / (np.std(f) + 1e-8)
            obs = np.concatenate(([equity], w, f))
        else:
            obs = np.concatenate(([equity], w))

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        """
        Advance the environment one step given an action.

        Parameters
        ----------
        action : ndarray
            Target portfolio weights to apply.

        Returns
        -------
        tuple
            observation : ndarray
                Updated observation vector.
            reward : float
                Scalar reward signal from the core environment.
            terminated : bool
                Whether the episode has reached its end.
            truncated : bool
                Whether the episode was truncated externally.
            info : dict
                Diagnostic and performance metrics.
        """
        obs_core, reward, done, info = self.core.step(action)

        if not np.isfinite(reward):
            print(f"[Warning] Non-finite reward at t={self.core.t}: {reward}")
            reward = 0.0

        obs = self._build_obs(obs_core)

        if np.isnan(obs).any():
            print(f"[NaN detected in obs] t={self.core.t}, replacing with zeros.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(reward):
            reward = 0.0

        obs = np.clip(obs, -1e6, 1e6)
        reward = float(np.clip(reward, -1e3, 1e3))

        terminated = bool(done)
        truncated = False

        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """
        Print a simple textual representation of the current simulation state.

        Displays the current step index and portfolio equity for quick debugging.
        """
        print(f"Step {self.core.t}, Equity {self.core.equity:.2f}")
