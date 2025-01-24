# Momentum Backtest

`momentum_backtest.py`, that demonstrates a straightforward workflow for quantitative strategy research. The script covers data generation, signal calculation, backtesting, and performance analysis, making it an excellent starting point for exploring momentum-based strategies.

## Purpose

The script implements the following steps:

### 1. Data Generation
- Simulates daily price data for multiple synthetic assets using a simple random-walk (geometric Brownian motion-style) process.
- Ensures reproducibility by setting a fixed random seed.

### 2. Signal Calculation
- Computes short-term and long-term moving averages (MAs) for each asset.
- Generates binary momentum signals:
  - **1**: If the short-term MA > long-term MA.
  - **0**: Otherwise.

### 3. Backtest Execution
- Calculates daily returns from the synthetic prices.
- Simulates a trading strategy by applying signals (with a one-day shift), assuming full allocation to any asset that triggers a "long" signal.
- Aggregates returns to produce the overall strategy performance.

### 4. Performance Metrics
- Computes and prints common metrics for strategy analysis:
  - **Cumulative Return**
  - **Annualized Return**
  - **Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**

### 5. Visualization
- Plots a sample asset's price trajectory.
- Plots the strategy's cumulative returns over the backtesting period.