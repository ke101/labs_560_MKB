import numpy as np
import pandas as pd
from typing import Tuple, List
from abc import ABC, abstractmethod


class TradingAlgorithm(ABC):
    """Base class for trading algorithms"""

    @abstractmethod
    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # Return signals: 1=buy, -1=sell, 0=hold
        pass


class MovingAverageCrossover(TradingAlgorithm):
    """
    MA crossover strategy
    Buy when short MA crosses above long MA (golden cross)
    Sell when short MA crosses below long MA (death cross)
    """

    def __init__(self, short_window=20, long_window=50, use_ema=False):
        self.short_window = short_window
        self.long_window = long_window
        self.use_ema = use_ema
        self.name = "MA_Crossover"

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'price': prices})

        if self.use_ema:
            df['short_ma'] = prices.ewm(span=self.short_window, adjust=False).mean()
            df['long_ma'] = prices.ewm(span=self.long_window, adjust=False).mean()
        else:
            df['short_ma'] = prices.rolling(window=self.short_window).mean()
            df['long_ma'] = prices.rolling(window=self.long_window).mean()

        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        df = self.calculate_indicators(prices)
        signals = pd.Series(0, index=prices.index)

        # position=1 when short MA > long MA, else 0
        df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        df['signal'] = df['position'].diff()

        signals[df['signal'] == 1] = 1    # golden cross buy
        signals[df['signal'] == -1] = -1  # death cross sell

        return signals

    def get_parameters(self):
        return {
            'algorithm': 'Moving Average Crossover',
            'short_window': self.short_window,
            'long_window': self.long_window,
            'ma_type': 'EMA' if self.use_ema else 'SMA'
        }


class RSIStrategy(TradingAlgorithm):
    """
    RSI strategy - Relative Strength Index
    RSI < oversold (default 30) -> oversold, buy
    RSI > overbought (default 70) -> overbought, sell
    """

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = "RSI"

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI, RSI = 100 - 100/(1+RS), RS = avg gain/avg loss"""
        delta = prices.diff()

        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)

        # Calculate avg gain/loss with EWM (smoothed RSI)
        avg_gain = gains.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.replace([np.inf, -np.inf], np.nan)

        return rsi

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'price': prices})
        df['rsi'] = self.calculate_rsi(prices)
        df['oversold_line'] = self.oversold
        df['overbought_line'] = self.overbought
        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        df = self.calculate_indicators(prices)
        signals = pd.Series(0, index=prices.index)

        prev_rsi = df['rsi'].shift(1)
        cur_rsi = df['rsi']

        # RSI crosses below oversold -> buy
        buy_cond = (prev_rsi >= self.oversold) & (cur_rsi < self.oversold)
        signals[buy_cond] = 1

        # RSI crosses above overbought -> sell
        sell_cond = (prev_rsi <= self.overbought) & (cur_rsi > self.overbought)
        signals[sell_cond] = -1

        return signals

    def get_parameters(self):
        return {
            'algorithm': 'RSI Strategy',
            'period': self.period,
            'oversold_threshold': self.oversold,
            'overbought_threshold': self.overbought
        }


class HybridStrategy(TradingAlgorithm):
    """
    Hybrid strategy: MA + RSI
    confirm mode: execute only when MA signals and RSI confirms (not in extreme zones)
    any mode: execute when either indicator signals
    """

    def __init__(self, ma_short=20, ma_long=50, rsi_period=14,
                 rsi_oversold=30, rsi_overbought=70, mode='confirm'):
        self.ma_strategy = MovingAverageCrossover(ma_short, ma_long)
        self.rsi_strategy = RSIStrategy(rsi_period, rsi_oversold, rsi_overbought)
        self.mode = mode
        self.name = "Hybrid_MA_RSI"

        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        ma_ind = self.ma_strategy.calculate_indicators(prices)
        rsi_ind = self.rsi_strategy.calculate_indicators(prices)

        df = ma_ind.copy()
        df['rsi'] = rsi_ind['rsi']
        df['oversold_line'] = rsi_ind['oversold_line']
        df['overbought_line'] = rsi_ind['overbought_line']
        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        df = self.calculate_indicators(prices)
        ma_signals = self.ma_strategy.generate_signals(prices)
        rsi_signals = self.rsi_strategy.generate_signals(prices)

        signals = pd.Series(0, index=prices.index)

        if self.mode == 'confirm':
            # MA buy signal and RSI not overbought -> buy
            buy_cond = (ma_signals == 1) & (df['rsi'] < self.rsi_overbought)
            sell_cond = (ma_signals == -1) & (df['rsi'] > self.rsi_oversold)
            signals[buy_cond] = 1
            signals[sell_cond] = -1

        elif self.mode == 'any':
            buy_cond = (ma_signals == 1) | (rsi_signals == 1)
            sell_cond = (ma_signals == -1) | (rsi_signals == -1)
            signals[buy_cond] = 1
            signals[sell_cond] = -1

        return signals

    def get_parameters(self):
        return {
            'algorithm': 'Hybrid Strategy (MA + RSI)',
            'mode': self.mode,
            'ma_short_window': self.ma_short,
            'ma_long_window': self.ma_long,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought
        }


class AlgorithmEvaluator:
    """Evaluate trading algorithm performance, calculate MAE/RMSE metrics"""

    @staticmethod
    def mean_absolute_error(actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        return np.mean(np.abs(actual - predicted))

    @staticmethod
    def root_mean_squared_error(actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        return np.sqrt(np.mean((actual - predicted) ** 2))

    @staticmethod
    def mean_absolute_percentage_error(actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        mask = actual != 0  # avoid division by zero
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    @staticmethod
    def direction_accuracy(actual_returns, predicted_signals):
        """Direction accuracy: percentage of correct direction predictions"""
        actual_returns = np.array(actual_returns)
        predicted_signals = np.array(predicted_signals)

        actual_dir = np.sign(actual_returns)
        mask = predicted_signals != 0  # only consider actual trades

        if np.sum(mask) == 0:
            return 0.0

        correct = np.sum(actual_dir[mask] == predicted_signals[mask])
        return (correct / np.sum(mask)) * 100

    @staticmethod
    def calculate_all_metrics(actual, predicted):
        return {
            'MAE': AlgorithmEvaluator.mean_absolute_error(actual, predicted),
            'RMSE': AlgorithmEvaluator.root_mean_squared_error(actual, predicted),
            'MAPE': AlgorithmEvaluator.mean_absolute_percentage_error(actual, predicted)
        }


def compare_algorithms(prices, algorithms):
    """Compare multiple algorithms on same price data"""
    results = pd.DataFrame({'price': prices})

    for algo in algorithms:
        signals = algo.generate_signals(prices)
        results[f'{algo.name}_signal'] = signals

        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        print(f"{algo.name}: {buy_count} buy, {sell_count} sell")

    return results


def optimize_ma_parameters(prices, short_range=range(5, 30, 5),
                           long_range=range(30, 100, 10)):
    """Brute force search for optimal short/long window parameters in MA crossover"""
    best_return = float('-inf')
    best_params = (20, 50)

    for short in short_range:
        for long in long_range:
            if short >= long:
                continue

            algo = MovingAverageCrossover(short, long)
            signals = algo.generate_signals(prices)
            returns = prices.pct_change()
            strat_ret = (signals.shift(1) * returns).sum()

            if strat_ret > best_return:
                best_return = strat_ret
                best_params = (short, long)

    return best_params[0], best_params[1], best_return


if __name__ == "__main__":
    # Generate synthetic stock price data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 200))
    prices = pd.Series(trend + noise + seasonal, index=dates)

    print("=" * 60)
    print("Trading Algorithm Demo")
    print("=" * 60)

    # Test MA Crossover
    print("\n1. MA Crossover")
    print("-" * 40)
    ma_algo = MovingAverageCrossover(short_window=10, long_window=30)
    ma_signals = ma_algo.generate_signals(prices)
    print(f"Params: {ma_algo.get_parameters()}")
    print(f"Buy: {(ma_signals == 1).sum()}, Sell: {(ma_signals == -1).sum()}")

    # Test RSI
    print("\n2. RSI Strategy")
    print("-" * 40)
    rsi_algo = RSIStrategy(period=14, oversold=30, overbought=70)
    rsi_signals = rsi_algo.generate_signals(prices)
    print(f"Params: {rsi_algo.get_parameters()}")
    print(f"Buy: {(rsi_signals == 1).sum()}, Sell: {(rsi_signals == -1).sum()}")

    # Test Hybrid Strategy
    print("\n3. Hybrid (MA + RSI)")
    print("-" * 40)
    hybrid_algo = HybridStrategy(mode='confirm')
    hybrid_signals = hybrid_algo.generate_signals(prices)
    print(f"Params: {hybrid_algo.get_parameters()}")
    print(f"Buy: {(hybrid_signals == 1).sum()}, Sell: {(hybrid_signals == -1).sum()}")

    # Evaluation Metrics
    print("\n4. Evaluation Metrics")
    print("-" * 40)
    evaluator = AlgorithmEvaluator()
    ma_indicators = ma_algo.calculate_indicators(prices)
    predicted = ma_indicators['short_ma'].dropna()
    actual = prices.loc[predicted.index]

    metrics = evaluator.calculate_all_metrics(actual.values, predicted.values)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    print("\nDone.")
