#
#  momentum-backtest.py
#  Created by Bruno on 10/14/21.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Replace this method with actual data
def generate_synthetic_prices(num_assets=5, start_date='2020-01-01', end_date='2021-12-31', seed=42):
    """
    Generate synthetic daily price data for multiple assets using
    a simplified random-walk (geometric Brownian motion-like) approach.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Parameters for the random walk
    drift = 0.0002   # daily drift
    volatility = 0.01
    
    prices_dict = {}
    for asset in range(num_assets):
        # Start all assets at different base prices to differentiate them
        base_price = 50 + asset * 10
        
        # Random daily returns
        daily_returns = np.random.normal(drift, volatility, n_days)
        
        # Convert returns to price via a cumulative product
        price_series = base_price * np.exp(np.cumsum(daily_returns))
        
        prices_dict[f"Asset_{asset+1}"] = price_series
    
    prices_df = pd.DataFrame(prices_dict, index=dates)
    return prices_df

def calculate_moving_average_signals(prices, short_window=20, long_window=50):
    """
    For each asset, generate a trading signal based on a short-term
    moving average (MA) crossing above/below a long-term MA.
    Signal = 1 if short MA > long MA, else 0.
    """
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    
    # Generate signal (1 when short_ma > long_ma, else 0)
    signals = (short_ma > long_ma).astype(int)
    return signals

def backtest_signals(prices, signals):
    """
    A simple backtest assuming:
      - We invest in each asset with weight proportional to its signal.
      - If signal = 1, we go fully invested in that asset; if signal = 0, no position.
      - We do not use leverage (weights sum up if multiple signals are 1).
      - Daily returns are derived from daily price changes.
    """
    # Daily returns (percent change)
    daily_returns = prices.pct_change().fillna(0)
    
    # Position each day: signals * 1 (meaning we hold the asset fully if signal=1, else 0)
    # For a real strategy, you might want to normalize weights if multiple signals are 1, etc.
    positions = signals
    
    # Strategy returns = sum of (positions * daily_returns) across all assets
    # This is effectively the "portfolio return" if you take equal notional positions for each asset
    strategy_returns = (positions.shift(1) * daily_returns).mean(axis=1)  # shift(1) to simulate entering positions at close
    
    # Accumulate strategy returns into an index (1 + r1)*(1 + r2)*... - 1
    cumulative_return = (1 + strategy_returns).cumprod() - 1
    
    return strategy_returns, cumulative_return

def calculate_performance_metrics(strategy_returns):
    """
    Compute common performance metrics such as:
    - Cumulative return
    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown
    """
    # Assuming daily data
    days_per_year = 252
    
    # Cumulative return
    cumulative_return = (1 + strategy_returns).prod() - 1
    
    # Annualized return (approx)
    total_days = len(strategy_returns)
    annualized_return = (1 + cumulative_return) ** (days_per_year / total_days) - 1
    
    # Annualized volatility
    annualized_volatility = strategy_returns.std() * np.sqrt(days_per_year)
    
    # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    
    # Maximum drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def main():
    # 1. Generate Synthetic Data
    prices_df = generate_synthetic_prices(num_assets=5, 
                                          start_date='2020-01-01', 
                                          end_date='2021-12-31', 
                                          seed=42)
    
    # 2. Calculate Signals (momentum-based using MAs)
    signals_df = calculate_moving_average_signals(prices_df, 
                                                  short_window=20, 
                                                  long_window=50)
    
    # 3. Backtest the Strategy
    strategy_returns, cumulative_return_series = backtest_signals(prices_df, signals_df)
    
    # 4. Calculate and Print Performance Metrics
    metrics = calculate_performance_metrics(strategy_returns)
    print("Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 5. Plot Results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # Plot one of the synthetic assets
    prices_df["Asset_1"].plot(ax=axes[0], title="Asset_1 Price")
    axes[0].set_ylabel("Price")

    # Plot the cumulative return of our strategy
    cumulative_return_series.plot(ax=axes[1], title="Strategy Cumulative Return")
    axes[1].set_ylabel("Cumulative Return (%)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
