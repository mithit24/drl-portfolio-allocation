
import numpy as np
import gymnasium as gym

class PortfolioGym(gym.Env):
    def __init__(self, features, prices, tickers, window=60, cost=1e-3):
        super().__init__()
        self.features = features.copy()
        self.prices = prices.copy()
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.window = window
        self.cost = cost

        # Clean up features
        self.features = self.features.replace([np.inf, -np.inf], np.nan).ffill(limit=5).fillna(0)
        self.features = self.features.astype(np.float32)

        # Compute returns
        self.rets = self.prices.pct_change(fill_method=None).shift(-1).fillna(0).astype(np.float32)
        self.valid_idx = self.features.index.intersection(self.rets.index)
        self.i = self.window
        self.done = False

        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.nav = 1.0

        # Spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = np.random.randint(0, len(self.features) - 200)  # random start
        self.nav = 1.0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        obs = self.features.iloc[self.i].to_numpy(dtype=np.float32)
        return obs, {}


    def step(self, action):
        # Softmax portfolio weights (with numerical stability)
        exp_a = np.exp(action - np.max(action))
        w_target = exp_a / (exp_a.sum() + 1e-8)

        # Trading cost
        turnover = np.abs(w_target - self.weights).sum()
        tc = self.cost * turnover

        # Portfolio return
        port_ret = np.dot(self.rets.iloc[self.i].values, self.weights)
        vol_proxy = float(
            np.mean(self.features.iloc[self.i].xs("rv_10", axis=0, level=1))  # use available feature
        )

        # Update NAV and weights
        self.nav *= (1 + port_ret - tc)
        self.weights = w_target
        self.i += 1

        # Reward: risk-adjusted return (Sharpe-like)
        reward = (port_ret - tc) / (np.sqrt(vol_proxy) + 1e-6)
        reward = np.clip(reward, -5, 5)  # stability

        done = self.i >= len(self.features) - 2
        obs = self.features.iloc[self.i].to_numpy(dtype=np.float32)
        info = {"nav": self.nav, "turnover": turnover, "reward": reward}

        return obs, float(reward), done, False, info
