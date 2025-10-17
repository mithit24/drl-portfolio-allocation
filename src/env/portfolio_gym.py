import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.env.portfolio_env import PortfolioEnv

class PortfolioGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, feature_df=None, start_equity=1_000_000):
        super().__init__()
        self.core = PortfolioEnv(prices, start_equity=start_equity)
        self.feature_df = feature_df
        self.tickers = self.core.tickers
        self.n_assets = len(self.tickers)

        # Observation: equity + weights + optional features
        # If no features, we include equity and weights only
        feat_dim = 0 if feature_df is None else feature_df.shape[1]
        obs_dim = 1 + self.n_assets + feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: target weights (0 to 1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_core = self.core.reset()
        obs = self._build_obs(obs_core)
        return obs.astype(np.float32), {}

    def _build_obs(self, core_obs):
        equity = float(core_obs["equity"])
        w = np.array(core_obs["weights"], dtype=np.float32)


        if self.feature_df is not None:
            # --- Safe timestep alignment ---
            t = min(self.core.t, len(self.feature_df) - 1)
            f = self.feature_df.iloc[t].to_numpy(dtype=np.float32)
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            f = (f - np.mean(f)) / (np.std(f) + 1e-8)
            obs = np.concatenate(([equity], w, f))
        else:
            obs = np.concatenate(([equity], w))

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)





    def step(self, action):
        obs_core, reward, done, info = self.core.step(action)

        # Clamp reward for stability (avoid inf/nan propagation)
        if not np.isfinite(reward):
            print(f"[Warning] Non-finite reward at t={self.core.t}: {reward}")
            reward = 0.0

        obs = self._build_obs(obs_core)

        # --- Full NaN guard ---
        if np.isnan(obs).any():
            print(f"[NaN detected in obs] t={self.core.t}, replacing with zeros.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(reward):
            reward = 0.0

        # Enforce numeric stability
        obs = np.clip(obs, -1e6, 1e6)
        reward = float(np.clip(reward, -1e3, 1e3))

        # --- ✅ New Gymnasium API ---
        terminated = bool(done)
        truncated = False  # no early cutoff



        return obs.astype(np.float32), reward, terminated, truncated, info







    def render(self):
        print(f"Step {self.core.t}, Equity {self.core.equity:.2f}")
