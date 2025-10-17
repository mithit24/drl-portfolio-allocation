# drl-portfolio-allocation

# DRL Portfolio Allocation

This project implements a **Deep Reinforcement Learning (DRL)** framework for multi-asset portfolio allocation using high-frequency ETF data. The system trains a Soft Actor–Critic (SAC) agent to allocate capital dynamically across multiple assets, with realistic trading frictions and feature engineering.

## 📌 Key Features
- **Data Pipeline**: Automated download and cleaning of 2-minute ETF data from Yahoo Finance.  
- **Feature Engineering**: Momentum, volatility, RSI, MACD, Z-scores, and volume dynamics built per asset.  
- **Custom Gym Environment**: `PortfolioEnv` and `PortfolioGym` simulate realistic portfolio trading with:
  - Execution costs
  - Volume participation constraints
  - Risk penalties and turnover costs
- **RL Training**: Soft Actor–Critic agent (Stable-Baselines3) trained with vectorized environments.  
- **Evaluation & Benchmarking**: Compare against SPY and equal-weight portfolios, compute Sharpe ratios, and visualize equity curves.

---

## 🧱 Project Structure

```plaintext
.
├── data/                        # Raw and processed data (not tracked by git)
├── logs/                        # TensorBoard logs and evals (ignored in .gitignore)
├── models/                      # Trained model weights (ignored in .gitignore)
├── src/
│   ├── env/
│   │   ├── portfolio_env.py     # Core portfolio trading environment
│   │   └── portfolio_gym.py     # Gym wrapper for RL training
│   └── utils/
│       ├── costs.py             # Execution cost & volatility functions
│       └── features.py          # Feature engineering pipeline
├── SAC_training.ipynb           # Main training notebook
├── train_sac.ipynb              # Script-style training alternative
├── feature_engineering.ipynb    # Data & feature generation
├── README.md
└── requirements.txt
