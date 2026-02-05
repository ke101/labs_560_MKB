import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from trading_algorithm import (
    MovingAverageCrossover,
    RSIStrategy,
    HybridStrategy,
    AlgorithmEvaluator,
    TradingAlgorithm
)


# ============================================================================
# Core Signal Generation Functions
# ============================================================================

def generate_ma_signals(prices: pd.Series,
                        short_window: int = 20,
                        long_window: int = 50,
                        use_ema: bool = False) -> Dict:
    """
    Generate trading signals using Moving Average Crossover strategy.

    Args:
        prices: Series of stock prices (closing prices recommended)
        short_window: Period for short-term MA (default: 20)
        long_window: Period for long-term MA (default: 50)
        use_ema: Use Exponential MA instead of Simple MA

    Returns:
        Dictionary containing:
            - 'signals': Series of buy(1)/sell(-1)/hold(0) signals
            - 'indicators': DataFrame with price and MA values
            - 'summary': Dict with signal counts and parameters
    """
    algo = MovingAverageCrossover(short_window, long_window, use_ema)
    signals = algo.generate_signals(prices)
    indicators = algo.calculate_indicators(prices)

    return {
        'signals': signals,
        'indicators': indicators,
        'summary': {
            'algorithm': 'Moving Average Crossover',
            'parameters': algo.get_parameters(),
            'buy_signals': int((signals == 1).sum()),
            'sell_signals': int((signals == -1).sum()),
            'total_signals': int((signals != 0).sum())
        }
    }


def generate_rsi_signals(prices: pd.Series,
                         period: int = 14,
                         oversold: float = 30,
                         overbought: float = 70) -> Dict:
    """
    Generate trading signals using RSI strategy.

    Args:
        prices: Series of stock prices
        period: RSI calculation period (default: 14)
        oversold: Buy threshold - RSI below this indicates oversold (default: 30)
        overbought: Sell threshold - RSI above this indicates overbought (default: 70)

    Returns:
        Dictionary containing:
            - 'signals': Series of buy(1)/sell(-1)/hold(0) signals
            - 'indicators': DataFrame with price and RSI values
            - 'summary': Dict with signal counts and parameters
    """
    algo = RSIStrategy(period, oversold, overbought)
    signals = algo.generate_signals(prices)
    indicators = algo.calculate_indicators(prices)

    return {
        'signals': signals,
        'indicators': indicators,
        'summary': {
            'algorithm': 'RSI Strategy',
            'parameters': algo.get_parameters(),
            'buy_signals': int((signals == 1).sum()),
            'sell_signals': int((signals == -1).sum()),
            'total_signals': int((signals != 0).sum())
        }
    }


def generate_hybrid_signals(prices: pd.Series,
                            ma_short: int = 20,
                            ma_long: int = 50,
                            rsi_period: int = 14,
                            rsi_oversold: float = 30,
                            rsi_overbought: float = 70,
                            mode: str = 'confirm') -> Dict:
    """
    Generate trading signals using Hybrid (MA + RSI) strategy.

    Args:
        prices: Series of stock prices
        ma_short: Short MA period (default: 20)
        ma_long: Long MA period (default: 50)
        rsi_period: RSI calculation period (default: 14)
        rsi_oversold: RSI oversold threshold (default: 30)
        rsi_overbought: RSI overbought threshold (default: 70)
        mode: 'confirm' (both must agree) or 'any' (either can trigger)

    Returns:
        Dictionary containing:
            - 'signals': Series of buy(1)/sell(-1)/hold(0) signals
            - 'indicators': DataFrame with all indicator values
            - 'summary': Dict with signal counts and parameters
    """
    algo = HybridStrategy(ma_short, ma_long, rsi_period, rsi_oversold, rsi_overbought, mode)
    signals = algo.generate_signals(prices)
    indicators = algo.calculate_indicators(prices)

    return {
        'signals': signals,
        'indicators': indicators,
        'summary': {
            'algorithm': 'Hybrid Strategy (MA + RSI)',
            'parameters': algo.get_parameters(),
            'buy_signals': int((signals == 1).sum()),
            'sell_signals': int((signals == -1).sum()),
            'total_signals': int((signals != 0).sum())
        }
    }


# ============================================================================
# Multi-Stock Signal Generation
# ============================================================================

def generate_portfolio_signals(stock_data: Dict[str, pd.Series],
                               algorithm: str = 'hybrid',
                               **kwargs) -> Dict[str, Dict]:
    """
    Generate signals for multiple stocks in a portfolio.

    Args:
        stock_data: Dictionary mapping stock symbols to price series
                    e.g., {'AAPL': prices_aapl, 'GOOGL': prices_googl}
        algorithm: Algorithm to use - 'ma', 'rsi', or 'hybrid'
        **kwargs: Additional parameters passed to the algorithm

    Returns:
        Dictionary mapping stock symbols to their signal results
    """
    # Select algorithm function
    algo_functions = {
        'ma': generate_ma_signals,
        'rsi': generate_rsi_signals,
        'hybrid': generate_hybrid_signals
    }

    if algorithm not in algo_functions:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(algo_functions.keys())}")

    signal_func = algo_functions[algorithm]
    results = {}

    for symbol, prices in stock_data.items():
        try:
            results[symbol] = signal_func(prices, **kwargs)
            print(f"[{symbol}] Generated {results[symbol]['summary']['total_signals']} signals")
        except Exception as e:
            print(f"[{symbol}] Error generating signals: {e}")
            results[symbol] = None

    return results


# ============================================================================
# Signal Processing and Analysis
# ============================================================================

def get_signal_dates(signals: pd.Series) -> Dict[str, List]:
    """
    Extract dates of buy and sell signals.

    Args:
        signals: Series of trading signals

    Returns:
        Dictionary with 'buy_dates' and 'sell_dates' lists
    """
    return {
        'buy_dates': signals[signals == 1].index.tolist(),
        'sell_dates': signals[signals == -1].index.tolist()
    }


def signals_to_positions(signals: pd.Series) -> pd.Series:
    """
    Convert trading signals to position series.

    Signals: 1 (buy), -1 (sell), 0 (hold)
    Positions: 1 (long), 0 (no position)

    After a buy signal, position is 1 until a sell signal.
    """
    positions = pd.Series(0, index=signals.index)

    position = 0
    for i, signal in enumerate(signals):
        if signal == 1:
            position = 1
        elif signal == -1:
            position = 0
        positions.iloc[i] = position

    return positions


def calculate_signal_returns(prices: pd.Series,
                             signals: pd.Series,
                             transaction_cost: float = 0.001) -> Dict:
    """
    Calculate returns based on trading signals.

    Args:
        prices: Stock price series
        signals: Trading signal series
        transaction_cost: Cost per transaction as fraction (default: 0.1%)

    Returns:
        Dictionary with return metrics and series
    """
    # Convert signals to positions
    positions = signals_to_positions(signals)

    # Calculate daily returns
    daily_returns = prices.pct_change()

    # Strategy returns (only earn returns when in position)
    strategy_returns = positions.shift(1) * daily_returns

    # Deduct transaction costs
    signal_changes = signals.abs()
    transaction_costs = signal_changes * transaction_cost
    strategy_returns = strategy_returns - transaction_costs

    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    buy_hold_returns = (1 + daily_returns).cumprod()

    # Calculate metrics
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    buy_hold_return = buy_hold_returns.iloc[-1] - 1 if len(buy_hold_returns) > 0 else 0

    # Count trades
    num_trades = (signals != 0).sum()

    return {
        'daily_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'buy_hold_returns': buy_hold_returns,
        'total_return': float(total_return),
        'buy_hold_return': float(buy_hold_return),
        'excess_return': float(total_return - buy_hold_return),
        'num_trades': int(num_trades)
    }


# ============================================================================
# Algorithm Comparison
# ============================================================================

def compare_all_strategies(prices: pd.Series,
                           ma_params: Dict = None,
                           rsi_params: Dict = None,
                           hybrid_params: Dict = None) -> pd.DataFrame:
    """
    Compare all three strategies on the same price data.

    Args:
        prices: Stock price series
        ma_params: Parameters for MA strategy (optional)
        rsi_params: Parameters for RSI strategy (optional)
        hybrid_params: Parameters for Hybrid strategy (optional)

    Returns:
        DataFrame comparing strategy performance
    """
    # Default parameters
    ma_params = ma_params or {}
    rsi_params = rsi_params or {}
    hybrid_params = hybrid_params or {}

    # Generate signals for all strategies
    ma_result = generate_ma_signals(prices, **ma_params)
    rsi_result = generate_rsi_signals(prices, **rsi_params)
    hybrid_result = generate_hybrid_signals(prices, **hybrid_params)

    # Calculate returns for each
    ma_returns = calculate_signal_returns(prices, ma_result['signals'])
    rsi_returns = calculate_signal_returns(prices, rsi_result['signals'])
    hybrid_returns = calculate_signal_returns(prices, hybrid_result['signals'])

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Strategy': ['MA Crossover', 'RSI', 'Hybrid (MA+RSI)', 'Buy & Hold'],
        'Total Return (%)': [
            ma_returns['total_return'] * 100,
            rsi_returns['total_return'] * 100,
            hybrid_returns['total_return'] * 100,
            ma_returns['buy_hold_return'] * 100
        ],
        'Excess Return (%)': [
            ma_returns['excess_return'] * 100,
            rsi_returns['excess_return'] * 100,
            hybrid_returns['excess_return'] * 100,
            0.0
        ],
        'Num Trades': [
            ma_returns['num_trades'],
            rsi_returns['num_trades'],
            hybrid_returns['num_trades'],
            0
        ],
        'Buy Signals': [
            ma_result['summary']['buy_signals'],
            rsi_result['summary']['buy_signals'],
            hybrid_result['summary']['buy_signals'],
            1
        ],
        'Sell Signals': [
            ma_result['summary']['sell_signals'],
            rsi_result['summary']['sell_signals'],
            hybrid_result['summary']['sell_signals'],
            0
        ]
    })

    return comparison


def select_best_strategy(prices: pd.Series,
                         metric: str = 'total_return') -> Tuple[str, Dict]:
    """
    Automatically select the best performing strategy.

    Args:
        prices: Stock price series
        metric: Metric to use for comparison ('total_return' or 'excess_return')

    Returns:
        Tuple of (best_strategy_name, best_strategy_result)
    """
    strategies = {
        'MA Crossover': generate_ma_signals(prices),
        'RSI': generate_rsi_signals(prices),
        'Hybrid': generate_hybrid_signals(prices)
    }

    best_name = None
    best_result = None
    best_value = float('-inf')

    for name, result in strategies.items():
        returns = calculate_signal_returns(prices, result['signals'])
        value = returns[metric]

        if value > best_value:
            best_value = value
            best_name = name
            best_result = result

    print(f"Best strategy: {best_name} with {metric}={best_value:.4f}")
    return best_name, best_result


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_predictions(actual_prices: pd.Series,
                         predicted_prices: pd.Series) -> Dict:
    """
    Evaluate prediction accuracy using MAE, RMSE, and MAPE.

    Args:
        actual_prices: Actual stock prices
        predicted_prices: Predicted stock prices

    Returns:
        Dictionary with evaluation metrics
    """
    # Align indices
    common_idx = actual_prices.index.intersection(predicted_prices.index)
    actual = actual_prices.loc[common_idx].values
    predicted = predicted_prices.loc[common_idx].values

    evaluator = AlgorithmEvaluator()

    return {
        'MAE': evaluator.mean_absolute_error(actual, predicted),
        'RMSE': evaluator.root_mean_squared_error(actual, predicted),
        'MAPE': evaluator.mean_absolute_percentage_error(actual, predicted),
        'sample_size': len(common_idx)
    }


def evaluate_signal_accuracy(prices: pd.Series, signals: pd.Series) -> Dict:
    """
    Evaluate the accuracy of trading signals.

    Args:
        prices: Stock price series
        signals: Trading signal series

    Returns:
        Dictionary with signal accuracy metrics
    """
    # Calculate actual future returns
    future_returns = prices.pct_change().shift(-1)

    # Get indices where signals were generated
    signal_mask = signals != 0

    if signal_mask.sum() == 0:
        return {'accuracy': 0, 'profitable_trades': 0, 'total_trades': 0}

    # Check if signal direction matches return direction
    signal_at_trade = signals[signal_mask]
    return_at_trade = future_returns[signal_mask]

    correct_direction = (np.sign(signal_at_trade) == np.sign(return_at_trade)).sum()
    total_trades = signal_mask.sum()

    # Calculate profitable trades
    trade_returns = signal_at_trade * return_at_trade
    profitable = (trade_returns > 0).sum()

    return {
        'direction_accuracy': float(correct_direction / total_trades * 100) if total_trades > 0 else 0,
        'profitable_trades': int(profitable),
        'total_trades': int(total_trades),
        'win_rate': float(profitable / total_trades * 100) if total_trades > 0 else 0
    }


# ============================================================================
# Output Formatting Functions
# ============================================================================

def format_signal_report(result: Dict, symbol: str = "STOCK") -> str:
    """
    Format a signal result into a readable report.

    Args:
        result: Result dictionary from signal generation
        symbol: Stock symbol for labeling

    Returns:
        Formatted string report
    """
    summary = result['summary']

    report = f"""
================================================================================
Trading Signal Report: {symbol}
================================================================================
Algorithm: {summary['algorithm']}

Parameters:
{_format_params(summary['parameters'])}

Signal Summary:
    - Buy Signals:  {summary['buy_signals']}
    - Sell Signals: {summary['sell_signals']}
    - Total Signals: {summary['total_signals']}

Signal Interpretation:
    - 1  = BUY  (Enter long position)
    - -1 = SELL (Exit long position)
    - 0  = HOLD (No action)
================================================================================
"""
    return report


def _format_params(params: Dict) -> str:
    """Format parameters dictionary for display."""
    lines = []
    for key, value in params.items():
        if key != 'algorithm':
            lines.append(f"    - {key}: {value}")
    return '\n'.join(lines)


def export_signals_to_csv(signals: pd.Series,
                          indicators: pd.DataFrame,
                          filepath: str) -> None:
    """
    Export signals and indicators to CSV file.

    Args:
        signals: Trading signal series
        indicators: Indicator DataFrame
        filepath: Output file path
    """
    output = indicators.copy()
    output['signal'] = signals
    output['action'] = signals.map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
    output.to_csv(filepath)
    print(f"Signals exported to: {filepath}")


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    # Synthetic stock price with trend and volatility
    trend = np.linspace(100, 130, 200)
    noise = np.random.randn(200) * 3
    seasonal = 8 * np.sin(np.linspace(0, 6*np.pi, 200))
    prices = pd.Series(trend + noise + seasonal, index=dates)
    prices.name = 'close'

    print("=" * 70)
    print("Signal Generator Module - Demo")
    print("=" * 70)

    # Test MA signals
    print("\n1. Moving Average Crossover Signals")
    print("-" * 50)
    ma_result = generate_ma_signals(prices, short_window=10, long_window=30)
    print(format_signal_report(ma_result, "TEST_STOCK"))

    # Test RSI signals
    print("\n2. RSI Signals")
    print("-" * 50)
    rsi_result = generate_rsi_signals(prices, period=14)
    print(f"RSI Buy signals: {rsi_result['summary']['buy_signals']}")
    print(f"RSI Sell signals: {rsi_result['summary']['sell_signals']}")

    # Test Hybrid signals
    print("\n3. Hybrid Signals")
    print("-" * 50)
    hybrid_result = generate_hybrid_signals(prices, mode='confirm')
    print(f"Hybrid Buy signals: {hybrid_result['summary']['buy_signals']}")
    print(f"Hybrid Sell signals: {hybrid_result['summary']['sell_signals']}")

    # Compare all strategies
    print("\n4. Strategy Comparison")
    print("-" * 50)
    comparison = compare_all_strategies(prices)
    print(comparison.to_string(index=False))

    # Evaluate signal accuracy
    print("\n5. Signal Accuracy Evaluation")
    print("-" * 50)
    accuracy = evaluate_signal_accuracy(prices, hybrid_result['signals'])
    print(f"Direction Accuracy: {accuracy['direction_accuracy']:.1f}%")
    print(f"Win Rate: {accuracy['win_rate']:.1f}%")
    print(f"Profitable Trades: {accuracy['profitable_trades']}/{accuracy['total_trades']}")

    # Select best strategy
    print("\n6. Best Strategy Selection")
    print("-" * 50)
    best_name, best_result = select_best_strategy(prices)

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
