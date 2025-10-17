# drl-portfolio-allocation

# DRL Portfolio Allocation

This project implements a **Deep Reinforcement Learning (DRL)** framework for multi-asset portfolio allocation using high-frequency ETF data. The system trains a Soft Actor–Critic (SAC) agent to allocate capital dynamically across multiple assets, with realistic trading frictions and feature engineering.

##  Key Features
- **Data Pipeline**: Automated download and cleaning of 2-minute ETF data from Yahoo Finance.  
- **Feature Engineering**: Momentum, volatility, RSI, MACD, Z-scores, and volume dynamics built per asset.  
- **Custom Gym Environment**: `PortfolioEnv` and `PortfolioGym` simulate realistic portfolio trading with:
  - Execution costs
  - Volume participation constraints
  - Risk penalties and turnover costs
- **RL Training**: Soft Actor–Critic agent (Stable-Baselines3) trained with vectorized environments.  
- **Evaluation & Benchmarking**: Compare against SPY and equal-weight portfolios, compute Sharpe ratios, and visualize equity curves.

---

##  Project Structure

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
```
## Clone this Repository

git clone https://github.com/mithit24/drl-portfolio-allocation.git
cd drl-portfolio-allocation

## Create and activate this environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

## Install dependencies 

pip install -r requirements.txt

## Training 

To train the SAC agent on 2-minute ETF data:  jupyter notebook SAC_training.ipynb

or run the script version:python train_sac.ipynb

Training logs are saved under ./tb_sac_portfolio/ and can be viewed with:  tensorboard --logdir=./tb_sac_portfolio/

## Evaluation

After training, you can:

- Evaluate performance on a held-out time window 

- Benchmark against SPY and equal-weight strategies

- Plot equity curves and compute Sharpe ratios


**Example** : 

```plaintext
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy

model = SAC.load("models/sac_portfolio.zip")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)

print(f"Mean reward: {mean_reward:.4f}, Std: {std_reward:.4f}")

```
## Model Details

- Algorithm: Soft Actor–Critic (off-policy, entropy-regularized)

- Network: 2-layer MLP (512 units each)

- Buffer: 500,000 transitions

- Learning rate: 5e-5

- Discount factor: 0.999

- Action space: Continuous [0,1] target weights per asset

## Requirements

- Python ≥ 3.9

- pandas, numpy, matplotlib

- stable-baselines3

- gymnasium

- yfinance

- torch

### Author 
Mithit Sangwan
https://github.com/mithit24
