#
#  momentum-backtest.py
#  Created by Bruno on 5/9/22.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def generate_synthetic_prices(num_assets=1, start_date='2020-01-01', end_date='2021-12-31', seed=42):
    """
    Generate synthetic daily price data for multiple assets using
    a random-walk (geometric Brownian motion-like) approach.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Parameters for random walk
    drift = 0.0002
    volatility = 0.01
    
    prices_dict = {}
    for asset in range(num_assets):
        base_price = 50 + asset * 10
        # Random daily returns
        daily_returns = np.random.normal(drift, volatility, n_days)
        # Convert returns to price via cumulative product
        price_series = base_price * np.exp(np.cumsum(daily_returns))
        prices_dict[f"Asset_{asset+1}"] = price_series
    
    prices_df = pd.DataFrame(prices_dict, index=dates)
    return prices_df

def calculate_technical_features(prices, window_short=5, window_long=20):
    """
    Create a feature set using:
      - Past returns
      - Short/long moving averages
      - Momentum (price diff)
    
    You can expand this function with more sophisticated features (RSI, MACD, etc.).
    """
    df = prices.copy()
    
    # Daily returns
    df['daily_return'] = df['Asset_1'].pct_change().fillna(0)
    
    # Short and long moving averages
    df['ma_short'] = df['Asset_1'].rolling(window_short).mean()
    df['ma_long'] = df['Asset_1'].rolling(window_long).mean()
    
    # Momentum: difference between the current price and price X days ago
    df['momentum'] = df['Asset_1'] - df['Asset_1'].shift(window_short)
    
    # Clean up NaNs
    df.dropna(inplace=True)
    
    return df

def generate_labels(df, threshold=0.0):
    """
    Label next day's movement as 1 if the next day's return > threshold, else 0.
    By default, threshold=0.0 means we label 'up' if next day return is positive.
    """
    # Next day return
    df['next_return'] = df['daily_return'].shift(-1)
    
    # Binary label: 1 if next_return > threshold, else 0
    df['label'] = (df['next_return'] > threshold).astype(int)
    
    df.dropna(inplace=True)
    return df

def split_features_labels(df):
    """
    Separate the dataframe into features (X) and labels (y) for model training.
    """
    features = df[['daily_return', 'ma_short', 'ma_long', 'momentum']].values
    labels = df['label'].values
    return features, labels

def backtest(prices_df, df_with_predictions, signal_col='prediction'):
    """
    Backtest a strategy where:
      - If `prediction` = 1 (up), go long.
      - If `prediction` = 0 (down/flat), go to cash.
    
    Returns daily strategy returns and cumulative returns.
    """
    # We assume daily_return from prices_df is the actual daily return of Asset_1
    # The signal shifts to represent trading on the next day’s open or close:
    # For simplicity, we just apply the signal from the same day to the same day's return.
    # A more realistic approach would shift signals by 1 day.
    
    df_backtest = df_with_predictions.copy()
    df_backtest['signal'] = df_backtest[signal_col]
    
    # Strategy daily return = signal * daily return
    df_backtest['strategy_return'] = df_backtest['signal'] * prices_df['daily_return']
    
    # Cumulative return
    df_backtest['cumulative_return'] = (1 + df_backtest['strategy_return']).cumprod() - 1
    
    return df_backtest['strategy_return'], df_backtest['cumulative_return']

def calculate_performance_metrics(strategy_returns):
    """
    Compute performance metrics for the strategy:
      - Cumulative return
      - Annualized return
      - Annualized volatility
      - Sharpe ratio
      - Max drawdown
    """
    # Assuming ~252 trading days per year
    DAYS_PER_YEAR = 252
    
    cum_return = (1 + strategy_returns).prod() - 1
    total_days = len(strategy_returns)
    
    annualized_return = (1 + cum_return)**(DAYS_PER_YEAR / total_days) - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(DAYS_PER_YEAR)
    
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else np.nan
    
    # Max drawdown
    cum_prod = (1 + strategy_returns).cumprod()
    running_max = cum_prod.cummax()
    drawdown = (cum_prod - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Cumulative Return': cum_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    return metrics

def main():
    # 1. Generate synthetic prices
    prices = generate_synthetic_prices(num_assets=1,
                                       start_date='2020-01-01',
                                       end_date='2021-12-31',
                                       seed=42)
    
    # 2. Feature Engineering
    df = calculate_technical_features(prices)
    
    # 3. Generate Labels
    df = generate_labels(df, threshold=0.0)
    
    # 4. Split into features (X) and labels (y)
    X, y = split_features_labels(df)
    
    # We’ll use a simple train-test split (60% train, 40% test for illustration).
    # In practice, walk-forward validation or cross-validation is preferred.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    # 5. Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Generate Predictions on the Test Set
    y_pred = model.predict(X_test)
    
    # Evaluate predictive accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Out-of-sample Accuracy: {accuracy:.2f}")
    
    # 7. Insert the predictions back into the dataframe
    # Align the test predictions with the original dataframe indices
    df_test = df.iloc[len(X_train):].copy()
    df_test['prediction'] = y_pred
    
    # 8. Backtest using these predictions
    # Merge back the daily_return from df into df_test if needed
    df_test['daily_return'] = df.iloc[len(X_train):]['daily_return']
    
    strategy_returns, cum_returns = backtest(df_test, df_test, signal_col='prediction')
    
    # 9. Calculate and print performance metrics
    metrics = calculate_performance_metrics(strategy_returns)
    print("Performance Metrics (Test Period):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # 10. Plot Asset Price and Cumulative Strategy Returns
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    
    # Price of Asset_1 over the entire period
    prices['Asset_1'].plot(ax=axes[0], title='Synthetic Asset Price (Train + Test)')
    axes[0].set_ylabel('Price')
    
    # Strategy Cumulative Return during the test period
    cum_returns.plot(ax=axes[1], title='Strategy Cumulative Return (Test Period)')
    axes[1].set_ylabel('Cumulative Return')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
