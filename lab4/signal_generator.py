import pandas as pd
import numpy as np
from trading_algorithm import (
    MovingAverageCrossover, RSIStrategy, HybridStrategy,
    AlgorithmEvaluator, TradingAlgorithm
)


def generate_ma_signals(prices, short_window=20, long_window=50, use_ema=False):
    """Generate trading signals using MA crossover strategy"""
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


def generate_rsi_signals(prices, period=14, oversold=30, overbought=70):
    """Generate trading signals using RSI strategy"""
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


def generate_hybrid_signals(prices, ma_short=20, ma_long=50, rsi_period=14,
                            rsi_oversold=30, rsi_overbought=70, mode='confirm'):
    """Generate trading signals using hybrid strategy (MA+RSI)"""
    algo = HybridStrategy(ma_short, ma_long, rsi_period,
                          rsi_oversold, rsi_overbought, mode)
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


def generate_portfolio_signals(stock_data, algorithm='hybrid', **kwargs):
    """Generate signals for multiple stocks"""
    algo_funcs = {
        'ma': generate_ma_signals,
        'rsi': generate_rsi_signals,
        'hybrid': generate_hybrid_signals
    }

    if algorithm not in algo_funcs:
        raise ValueError(f"Unknown algorithm: {algorithm}, options: {list(algo_funcs.keys())}")

    func = algo_funcs[algorithm]
    results = {}

    for symbol, prices in stock_data.items():
        try:
            results[symbol] = func(prices, **kwargs)
            print(f"[{symbol}] {results[symbol]['summary']['total_signals']} signals generated")
        except Exception as e:
            print(f"[{symbol}] Error: {e}")
            results[symbol] = None

    return results


# ---------- Signal processing ----------

def get_signal_dates(signals):
    """Extract dates for buy/sell signals"""
    return {
        'buy_dates': signals[signals == 1].index.tolist(),
        'sell_dates': signals[signals == -1].index.tolist()
    }


def signals_to_positions(signals):
    """
    Convert signals to positions: hold after buy (1), no position after sell (0)
    """
    positions = pd.Series(0, index=signals.index)
    pos = 0
    for i, sig in enumerate(signals):
        if sig == 1:
            pos = 1
        elif sig == -1:
            pos = 0
        positions.iloc[i] = pos
    return positions


def calculate_signal_returns(prices, signals, transaction_cost=0.001):
    """Calculate strategy returns based on signals, deduct transaction costs"""
    positions = signals_to_positions(signals)

    daily_ret = prices.pct_change()

    # only earn returns when holding position
    strat_ret = positions.shift(1) * daily_ret

    # deduct transaction costs
    cost = signals.abs() * transaction_cost
    strat_ret = strat_ret - cost

    cum_ret = (1 + strat_ret).cumprod()
    bh_ret = (1 + daily_ret).cumprod()

    total = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    bh_total = bh_ret.iloc[-1] - 1 if len(bh_ret) > 0 else 0
    n_trades = (signals != 0).sum()

    return {
        'daily_returns': strat_ret,
        'cumulative_returns': cum_ret,
        'buy_hold_returns': bh_ret,
        'total_return': float(total),
        'buy_hold_return': float(bh_total),
        'excess_return': float(total - bh_total),
        'num_trades': int(n_trades)
    }


# ---------- Strategy comparison ----------

def compare_all_strategies(prices, ma_params=None, rsi_params=None, hybrid_params=None):
    """Compare performance of three strategies on same data"""
    ma_params = ma_params or {}
    rsi_params = rsi_params or {}
    hybrid_params = hybrid_params or {}

    ma_result = generate_ma_signals(prices, **ma_params)
    rsi_result = generate_rsi_signals(prices, **rsi_params)
    hybrid_result = generate_hybrid_signals(prices, **hybrid_params)

    ma_ret = calculate_signal_returns(prices, ma_result['signals'])
    rsi_ret = calculate_signal_returns(prices, rsi_result['signals'])
    hybrid_ret = calculate_signal_returns(prices, hybrid_result['signals'])

    comparison = pd.DataFrame({
        'Strategy': ['MA Crossover', 'RSI', 'Hybrid (MA+RSI)', 'Buy & Hold'],
        'Total Return (%)': [
            ma_ret['total_return'] * 100,
            rsi_ret['total_return'] * 100,
            hybrid_ret['total_return'] * 100,
            ma_ret['buy_hold_return'] * 100
        ],
        'Excess Return (%)': [
            ma_ret['excess_return'] * 100,
            rsi_ret['excess_return'] * 100,
            hybrid_ret['excess_return'] * 100,
            0.0
        ],
        'Num Trades': [
            ma_ret['num_trades'], rsi_ret['num_trades'],
            hybrid_ret['num_trades'], 0
        ],
        'Buy Signals': [
            ma_result['summary']['buy_signals'],
            rsi_result['summary']['buy_signals'],
            hybrid_result['summary']['buy_signals'], 1
        ],
        'Sell Signals': [
            ma_result['summary']['sell_signals'],
            rsi_result['summary']['sell_signals'],
            hybrid_result['summary']['sell_signals'], 0
        ]
    })
    return comparison


def select_best_strategy(prices, metric='total_return'):
    """Automatically select best performing strategy"""
    strategies = {
        'MA Crossover': generate_ma_signals(prices),
        'RSI': generate_rsi_signals(prices),
        'Hybrid': generate_hybrid_signals(prices)
    }

    best_name, best_result = None, None
    best_val = float('-inf')

    for name, result in strategies.items():
        ret = calculate_signal_returns(prices, result['signals'])
        val = ret[metric]
        if val > best_val:
            best_val = val
            best_name = name
            best_result = result

    print(f"Best: {best_name} ({metric}={best_val:.4f})")
    return best_name, best_result


# ---------- Evaluation ----------

def evaluate_predictions(actual_prices, predicted_prices):
    """Evaluate prediction accuracy using MAE/RMSE/MAPE"""
    common_idx = actual_prices.index.intersection(predicted_prices.index)
    actual = actual_prices.loc[common_idx].values
    predicted = predicted_prices.loc[common_idx].values

    ev = AlgorithmEvaluator()
    return {
        'MAE': ev.mean_absolute_error(actual, predicted),
        'RMSE': ev.root_mean_squared_error(actual, predicted),
        'MAPE': ev.mean_absolute_percentage_error(actual, predicted),
        'sample_size': len(common_idx)
    }


def evaluate_signal_accuracy(prices, signals):
    """Evaluate signal accuracy and win rate"""
    future_ret = prices.pct_change().shift(-1)
    mask = signals != 0

    if mask.sum() == 0:
        return {'accuracy': 0, 'profitable_trades': 0, 'total_trades': 0}

    sig = signals[mask]
    ret = future_ret[mask]

    correct = (np.sign(sig) == np.sign(ret)).sum()
    total = mask.sum()

    trade_pnl = sig * ret
    profitable = (trade_pnl > 0).sum()

    return {
        'direction_accuracy': float(correct / total * 100) if total > 0 else 0,
        'profitable_trades': int(profitable),
        'total_trades': int(total),
        'win_rate': float(profitable / total * 100) if total > 0 else 0
    }


# ---------- Output formatting ----------

def format_signal_report(result, symbol="STOCK"):
    """Format signal results into report"""
    s = result['summary']
    params_str = '\n'.join(f"    - {k}: {v}" for k, v in s['parameters'].items()
                           if k != 'algorithm')

    report = f"""
{'='*60}
Trading Signal Report: {symbol}
{'='*60}
Algorithm: {s['algorithm']}

Parameters:
{params_str}

Signal Summary:
    Buy:  {s['buy_signals']}
    Sell: {s['sell_signals']}
    Total: {s['total_signals']}
{'='*60}
"""
    return report


def export_signals_to_csv(signals, indicators, filepath):
    """Export signals and indicators to CSV"""
    output = indicators.copy()
    output['signal'] = signals
    output['action'] = signals.map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
    output.to_csv(filepath)
    print(f"Exported to {filepath}")


if __name__ == "__main__":
    # Simulated stock price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    trend = np.linspace(100, 130, 200)
    noise = np.random.randn(200) * 3
    seasonal = 8 * np.sin(np.linspace(0, 6*np.pi, 200))
    prices = pd.Series(trend + noise + seasonal, index=dates, name='close')

    print("=" * 60)
    print("Signal Generator - Demo")
    print("=" * 60)

    # MA signals
    print("\n1. MA Crossover")
    ma_result = generate_ma_signals(prices, short_window=10, long_window=30)
    print(format_signal_report(ma_result, "TEST"))

    # RSI signals
    print("2. RSI")
    rsi_result = generate_rsi_signals(prices, period=14)
    print(f"   Buy: {rsi_result['summary']['buy_signals']}, Sell: {rsi_result['summary']['sell_signals']}")

    # Hybrid signals
    print("\n3. Hybrid")
    hybrid_result = generate_hybrid_signals(prices, mode='confirm')
    print(f"   Buy: {hybrid_result['summary']['buy_signals']}, Sell: {hybrid_result['summary']['sell_signals']}")

    # Strategy comparison
    print("\n4. Comparison")
    comp = compare_all_strategies(prices)
    print(comp.to_string(index=False))

    # Accuracy
    print("\n5. Accuracy")
    acc = evaluate_signal_accuracy(prices, hybrid_result['signals'])
    print(f"   Direction: {acc['direction_accuracy']:.1f}%")
    print(f"   Win Rate:  {acc['win_rate']:.1f}%")
    print(f"   Profitable: {acc['profitable_trades']}/{acc['total_trades']}")

    # Best strategy
    print("\n6. Best Strategy")
    best_name, best_result = select_best_strategy(prices)

    print("\nDone.")
