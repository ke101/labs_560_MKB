import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


class TradingAlgorithm(ABC):
    """Abstract base class for all trading algorithms."""

    @abstractmethod
    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate technical indicators for the given price series."""
        pass

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate trading signals based on price data.

        Returns:
            pd.Series: Signal series where:
                1 = Buy signal
                -1 = Sell signal
                0 = Hold (no action)
        """
        pass


class MovingAverageCrossover(TradingAlgorithm):
    """
    Moving Average Crossover Strategy
    ==================================
    A classic trend-following strategy that generates signals based on
    the crossover of short-term and long-term moving averages.

    Buy Signal: Short MA crosses above Long MA (Golden Cross)
    Sell Signal: Short MA crosses below Long MA (Death Cross)

    Parameters:
        short_window (int): Period for short-term moving average (default: 20)
        long_window (int): Period for long-term moving average (default: 50)
        use_ema (bool): Use Exponential MA instead of Simple MA (default: False)
    """

    def __init__(self, short_window: int = 20, long_window: int = 50, use_ema: bool = False):
        self.short_window = short_window
        self.long_window = long_window
        self.use_ema = use_ema
        self.name = "MA_Crossover"

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate short and long moving averages.

        Args:
            prices: Series of stock prices (typically closing prices)

        Returns:
            DataFrame with columns: ['price', 'short_ma', 'long_ma']
        """
        df = pd.DataFrame({'price': prices})

        if self.use_ema:
            # Exponential Moving Average - more weight on recent prices
            df['short_ma'] = prices.ewm(span=self.short_window, adjust=False).mean()
            df['long_ma'] = prices.ewm(span=self.long_window, adjust=False).mean()
        else:
            # Simple Moving Average
            df['short_ma'] = prices.rolling(window=self.short_window).mean()
            df['long_ma'] = prices.rolling(window=self.long_window).mean()

        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate buy/sell signals based on MA crossover.

        Logic:
            - Buy (1): When short MA crosses above long MA
            - Sell (-1): When short MA crosses below long MA
            - Hold (0): No crossover detected
        """
        df = self.calculate_indicators(prices)

        # Initialize signal series
        signals = pd.Series(0, index=prices.index)

        # Calculate crossover points
        # Position: 1 when short > long, 0 otherwise
        df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)

        # Signal occurs when position changes
        df['signal'] = df['position'].diff()

        # Buy signal: position changes from 0 to 1 (short crosses above long)
        signals[df['signal'] == 1] = 1

        # Sell signal: position changes from 1 to 0 (short crosses below long)
        signals[df['signal'] == -1] = -1

        return signals

    def get_parameters(self) -> dict:
        """Return algorithm parameters for documentation."""
        return {
            'algorithm': 'Moving Average Crossover',
            'short_window': self.short_window,
            'long_window': self.long_window,
            'ma_type': 'EMA' if self.use_ema else 'SMA'
        }


class RSIStrategy(TradingAlgorithm):
    """
    Relative Strength Index (RSI) Strategy
    =======================================
    A momentum oscillator that measures the speed and magnitude of price changes.
    RSI oscillates between 0 and 100.

    Buy Signal: RSI falls below oversold threshold (default: 30)
    Sell Signal: RSI rises above overbought threshold (default: 70)

    Parameters:
        period (int): RSI calculation period (default: 14)
        oversold (float): Oversold threshold for buy signals (default: 30)
        overbought (float): Overbought threshold for sell signals (default: 70)
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = "RSI"

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate the Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the period

        Args:
            prices: Series of stock prices

        Returns:
            Series of RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)

        # Calculate average gains and losses using exponential moving average
        # This is the standard "Smoothed RSI" method
        avg_gain = gains.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()

        # Calculate Relative Strength
        rs = avg_gain / avg_loss

        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases (division by zero)
        rsi = rsi.replace([np.inf, -np.inf], np.nan)

        return rsi

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate RSI indicator.

        Returns:
            DataFrame with columns: ['price', 'rsi', 'oversold_line', 'overbought_line']
        """
        df = pd.DataFrame({'price': prices})
        df['rsi'] = self.calculate_rsi(prices)
        df['oversold_line'] = self.oversold
        df['overbought_line'] = self.overbought

        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate buy/sell signals based on RSI levels.

        Logic:
            - Buy (1): RSI crosses below oversold threshold (stock is undervalued)
            - Sell (-1): RSI crosses above overbought threshold (stock is overvalued)
            - Hold (0): RSI is between thresholds
        """
        df = self.calculate_indicators(prices)

        # Initialize signal series
        signals = pd.Series(0, index=prices.index)

        # Track previous RSI state to detect crossovers
        prev_rsi = df['rsi'].shift(1)
        current_rsi = df['rsi']

        # Buy signal: RSI crosses from above to below oversold level
        buy_condition = (prev_rsi >= self.oversold) & (current_rsi < self.oversold)
        signals[buy_condition] = 1

        # Sell signal: RSI crosses from below to above overbought level
        sell_condition = (prev_rsi <= self.overbought) & (current_rsi > self.overbought)
        signals[sell_condition] = -1

        return signals

    def get_parameters(self) -> dict:
        """Return algorithm parameters for documentation."""
        return {
            'algorithm': 'RSI Strategy',
            'period': self.period,
            'oversold_threshold': self.oversold,
            'overbought_threshold': self.overbought
        }


class HybridStrategy(TradingAlgorithm):
    """
    Hybrid Strategy (MA Crossover + RSI)
    =====================================
    Combines Moving Average Crossover and RSI to filter signals and
    improve trading accuracy. Uses RSI as a confirmation indicator.

    Buy Signal: MA gives buy signal AND RSI is not overbought
    Sell Signal: MA gives sell signal AND RSI is not oversold

    This approach reduces false signals by requiring confirmation from
    both indicators before executing a trade.

    Parameters:
        ma_short (int): Short MA period (default: 20)
        ma_long (int): Long MA period (default: 50)
        rsi_period (int): RSI calculation period (default: 14)
        rsi_oversold (float): RSI oversold threshold (default: 30)
        rsi_overbought (float): RSI overbought threshold (default: 70)
        mode (str): Signal combination mode
                    'confirm' - Both indicators must agree
                    'any' - Either indicator can trigger
    """

    def __init__(self, ma_short: int = 20, ma_long: int = 50,
                 rsi_period: int = 14, rsi_oversold: float = 30,
                 rsi_overbought: float = 70, mode: str = 'confirm'):

        self.ma_strategy = MovingAverageCrossover(ma_short, ma_long)
        self.rsi_strategy = RSIStrategy(rsi_period, rsi_oversold, rsi_overbought)
        self.mode = mode
        self.name = "Hybrid_MA_RSI"

        # Store parameters
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def calculate_indicators(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate all indicators from both strategies.

        Returns:
            DataFrame with MA and RSI indicators combined
        """
        ma_indicators = self.ma_strategy.calculate_indicators(prices)
        rsi_indicators = self.rsi_strategy.calculate_indicators(prices)

        # Combine indicators
        df = ma_indicators.copy()
        df['rsi'] = rsi_indicators['rsi']
        df['oversold_line'] = rsi_indicators['oversold_line']
        df['overbought_line'] = rsi_indicators['overbought_line']

        return df

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        Generate signals using combined MA and RSI logic.

        Confirmation Mode (default):
            - Buy: MA buy signal + RSI < overbought (not overbought)
            - Sell: MA sell signal + RSI > oversold (not oversold)

        Any Mode:
            - Buy: MA buy signal OR RSI buy signal
            - Sell: MA sell signal OR RSI sell signal
        """
        df = self.calculate_indicators(prices)

        # Get individual signals
        ma_signals = self.ma_strategy.generate_signals(prices)
        rsi_signals = self.rsi_strategy.generate_signals(prices)

        # Initialize combined signal series
        signals = pd.Series(0, index=prices.index)

        if self.mode == 'confirm':
            # Confirmation mode: MA signal must be confirmed by RSI position
            # Buy: MA buy signal AND RSI not in overbought territory
            buy_condition = (ma_signals == 1) & (df['rsi'] < self.rsi_overbought)

            # Sell: MA sell signal AND RSI not in oversold territory
            sell_condition = (ma_signals == -1) & (df['rsi'] > self.rsi_oversold)

            signals[buy_condition] = 1
            signals[sell_condition] = -1

        elif self.mode == 'any':
            # Any mode: Either indicator can trigger a signal
            # Buy: Either MA or RSI gives buy signal
            buy_condition = (ma_signals == 1) | (rsi_signals == 1)

            # Sell: Either MA or RSI gives sell signal
            sell_condition = (ma_signals == -1) | (rsi_signals == -1)

            signals[buy_condition] = 1
            signals[sell_condition] = -1

        return signals

    def get_parameters(self) -> dict:
        """Return algorithm parameters for documentation."""
        return {
            'algorithm': 'Hybrid Strategy (MA + RSI)',
            'mode': self.mode,
            'ma_short_window': self.ma_short,
            'ma_long_window': self.ma_long,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought
        }


# ============================================================================
# Evaluation Metrics
# ============================================================================

class AlgorithmEvaluator:
    """
    Evaluator for trading algorithm performance.
    Calculates MAE, RMSE, and other relevant metrics.
    """

    @staticmethod
    def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        MAE = (1/n) * Σ|actual - predicted|

        Lower MAE indicates better prediction accuracy.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        return np.mean(np.abs(actual - predicted))

    @staticmethod
    def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        RMSE = sqrt((1/n) * Σ(actual - predicted)²)

        RMSE penalizes larger errors more heavily than MAE.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        return np.sqrt(np.mean((actual - predicted) ** 2))

    @staticmethod
    def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        MAPE = (100/n) * Σ|(actual - predicted) / actual|

        Expresses error as a percentage of actual values.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Avoid division by zero
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    @staticmethod
    def direction_accuracy(actual_returns: np.ndarray, predicted_signals: np.ndarray) -> float:
        """
        Calculate direction accuracy (hit rate).

        Measures how often the predicted signal correctly predicts
        the direction of price movement.

        Returns:
            float: Percentage of correct direction predictions (0-100)
        """
        actual_returns = np.array(actual_returns)
        predicted_signals = np.array(predicted_signals)

        # Get actual direction from returns
        actual_direction = np.sign(actual_returns)

        # Filter out hold signals (0) for fair comparison
        mask = predicted_signals != 0

        if np.sum(mask) == 0:
            return 0.0

        correct = np.sum(actual_direction[mask] == predicted_signals[mask])
        total = np.sum(mask)

        return (correct / total) * 100

    @staticmethod
    def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate all available metrics and return as dictionary."""
        return {
            'MAE': AlgorithmEvaluator.mean_absolute_error(actual, predicted),
            'RMSE': AlgorithmEvaluator.root_mean_squared_error(actual, predicted),
            'MAPE': AlgorithmEvaluator.mean_absolute_percentage_error(actual, predicted)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def compare_algorithms(prices: pd.Series,
                       algorithms: List[TradingAlgorithm]) -> pd.DataFrame:
    """
    Compare multiple trading algorithms on the same price data.

    Args:
        prices: Stock price series
        algorithms: List of TradingAlgorithm instances

    Returns:
        DataFrame with signals from each algorithm
    """
    results = pd.DataFrame({'price': prices})

    for algo in algorithms:
        signals = algo.generate_signals(prices)
        results[f'{algo.name}_signal'] = signals

        # Count signals
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        print(f"{algo.name}: {buy_count} buy signals, {sell_count} sell signals")

    return results


def optimize_ma_parameters(prices: pd.Series,
                           short_range: range = range(5, 30, 5),
                           long_range: range = range(30, 100, 10)) -> Tuple[int, int, float]:
    """
    Simple parameter optimization for MA Crossover strategy.
    Tests different combinations of short and long window periods.

    Args:
        prices: Stock price series
        short_range: Range of short window periods to test
        long_range: Range of long window periods to test

    Returns:
        Tuple of (best_short, best_long, best_return)
    """
    best_return = float('-inf')
    best_params = (20, 50)

    for short in short_range:
        for long in long_range:
            if short >= long:
                continue

            algo = MovingAverageCrossover(short, long)
            signals = algo.generate_signals(prices)

            # Calculate simple return based on signals
            returns = prices.pct_change()
            strategy_returns = (signals.shift(1) * returns).sum()

            if strategy_returns > best_return:
                best_return = strategy_returns
                best_params = (short, long)

    return best_params[0], best_params[1], best_return


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    # Generate synthetic stock price data with trend and noise
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 200))
    prices = pd.Series(trend + noise + seasonal, index=dates)

    print("=" * 60)
    print("Trading Algorithm Demo")
    print("=" * 60)

    # Test MA Crossover
    print("\n1. Moving Average Crossover Strategy")
    print("-" * 40)
    ma_algo = MovingAverageCrossover(short_window=10, long_window=30)
    ma_signals = ma_algo.generate_signals(prices)
    print(f"Parameters: {ma_algo.get_parameters()}")
    print(f"Buy signals: {(ma_signals == 1).sum()}")
    print(f"Sell signals: {(ma_signals == -1).sum()}")

    # Test RSI
    print("\n2. RSI Strategy")
    print("-" * 40)
    rsi_algo = RSIStrategy(period=14, oversold=30, overbought=70)
    rsi_signals = rsi_algo.generate_signals(prices)
    print(f"Parameters: {rsi_algo.get_parameters()}")
    print(f"Buy signals: {(rsi_signals == 1).sum()}")
    print(f"Sell signals: {(rsi_signals == -1).sum()}")

    # Test Hybrid Strategy
    print("\n3. Hybrid Strategy (MA + RSI)")
    print("-" * 40)
    hybrid_algo = HybridStrategy(mode='confirm')
    hybrid_signals = hybrid_algo.generate_signals(prices)
    print(f"Parameters: {hybrid_algo.get_parameters()}")
    print(f"Buy signals: {(hybrid_signals == 1).sum()}")
    print(f"Sell signals: {(hybrid_signals == -1).sum()}")

    # Evaluate with metrics
    print("\n4. Evaluation Metrics Demo")
    print("-" * 40)
    evaluator = AlgorithmEvaluator()

    # Create simple prediction for demo (using MA as predictor)
    ma_indicators = ma_algo.calculate_indicators(prices)
    predicted = ma_indicators['short_ma'].dropna()
    actual = prices.loc[predicted.index]

    metrics = evaluator.calculate_all_metrics(actual.values, predicted.values)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
