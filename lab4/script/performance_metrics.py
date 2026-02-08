from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt
from mock_trading import MockTradingEnvironment, SignalTrader
from signal_generator import generate_hybrid_signals



class PerformanceMetrics:
    """
    Compute performance metrics from an equity curve (portfolio value series).
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ):
        self.equity_curve = equity_curve.dropna().copy()
        self.equity_curve.index = pd.to_datetime(self.equity_curve.index)

        self.benchmark = None
        if benchmark is not None:
            b = benchmark.dropna().copy()
            b.index = pd.to_datetime(b.index)
            self.benchmark = b

        self.risk_free_rate = float(risk_free_rate)

        self.returns = self.equity_curve.pct_change().dropna()
        if self.benchmark is not None:
            self.benchmark_returns = self.benchmark.pct_change().dropna()

    def total_return_pct(self) -> float:
        start = float(self.equity_curve.iloc[0])
        end = float(self.equity_curve.iloc[-1])
        return (end / start - 1.0) * 100.0

    def annualized_return_pct(self, periods_per_year: int = 252) -> float:
        start = float(self.equity_curve.iloc[0])
        end = float(self.equity_curve.iloc[-1])
        total_growth = end / start

        n = len(self.equity_curve)
        years = n / float(periods_per_year)
        if years <= 0:
            return 0.0

        return (total_growth ** (1.0 / years) - 1.0) * 100.0

    def volatility_pct(self, periods_per_year: int = 252) -> float:
        if len(self.returns) == 0:
            return 0.0
        return float(self.returns.std() * np.sqrt(periods_per_year) * 100.0)

    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        if len(self.returns) == 0:
            return 0.0

        rf_per_period = self.risk_free_rate / float(periods_per_year)
        excess = self.returns - rf_per_period
        denom = excess.std()
        if denom == 0 or np.isnan(denom):
            return 0.0

        return float(np.sqrt(periods_per_year) * excess.mean() / denom)

    def max_drawdown_pct(self) -> Dict:
        curve = self.equity_curve
        running_max = curve.cummax()
        drawdown = (curve - running_max) / running_max * 100.0

        end_date = drawdown.idxmin()
        max_dd = float(drawdown.loc[end_date])

        start_date = curve.loc[:end_date].idxmax()
        return {
            "max_drawdown_pct": max_dd,
            "start_date": start_date,
            "end_date": end_date,
        }

    def compare_to_benchmark(self) -> Optional[Dict]:
        if self.benchmark is None:
            return None

        start = float(self.benchmark.iloc[0])
        end = float(self.benchmark.iloc[-1])
        bench_return = (end / start - 1.0) * 100.0

        strat_return = self.total_return_pct()
        return {
            "benchmark_total_return_pct": float(bench_return),
            "strategy_total_return_pct": float(strat_return),
            "excess_return_pct": float(strat_return - bench_return),
        }

    def get_all_metrics(self, periods_per_year: int = 252) -> Dict:
        metrics = {
            "final_portfolio_value": float(self.equity_curve.iloc[-1]),
            "total_return_pct": self.total_return_pct(),
            "annualized_return_pct": self.annualized_return_pct(periods_per_year),
            "volatility_pct": self.volatility_pct(periods_per_year),
            "sharpe_ratio": self.sharpe_ratio(periods_per_year),
        }
        dd = self.max_drawdown_pct()
        metrics.update(dd)

        bench = self.compare_to_benchmark()
        if bench is not None:
            metrics.update(bench)

        return metrics

    def print_metrics(self, periods_per_year: int = 252) -> None:
        m = self.get_all_metrics(periods_per_year)
        print("Performance metrics")
        print(f"Final portfolio value: {m['final_portfolio_value']:.6f}")
        print(f"Total return (%): {m['total_return_pct']:.6f}")
        print(f"Annualized return (%): {m['annualized_return_pct']:.6f}")
        print(f"Volatility (%): {m['volatility_pct']:.6f}")
        print(f"Sharpe ratio: {m['sharpe_ratio']:.6f}")
        print(f"Max drawdown (%): {m['max_drawdown_pct']:.6f}")
        print(f"Max drawdown start: {m['start_date']}")
        print(f"Max drawdown end: {m['end_date']}")

        if "benchmark_total_return_pct" in m:
            print(f"Benchmark total return (%): {m['benchmark_total_return_pct']:.6f}")
            print(f"Excess return (%): {m['excess_return_pct']:.6f}")

    def plot_equity_curve(self, benchmark_label: str = "Benchmark", save_path: Optional[str] = None) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve.values, label="Strategy", linewidth=2)

        if self.benchmark is not None:
            aligned = self.benchmark.reindex(self.equity_curve.index).dropna()
            plt.plot(aligned.index, aligned.values, label=benchmark_label, linewidth=2, alpha=0.7)

        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_drawdown(self, save_path: Optional[str] = None) -> None:
        curve = self.equity_curve
        running_max = curve.cummax()
        drawdown = (curve - running_max) / running_max * 100.0

        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
        plt.plot(drawdown.index, drawdown.values, linewidth=2)

        plt.title("Drawdown (%)")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def _self_test():

    df = pd.read_csv("AAPL_10y_1d.csv")
    date_col = None
    for c in df.columns:
        if c.lower() in ["date", "datetime", "timestamp"]:
            date_col = c
            break
    price_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["close", "adj close", "adj_close", "adjclose"]:
            price_col = c
            break

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    prices = df[price_col].astype(float)

    result = generate_hybrid_signals(prices, mode="confirm")
    signals = result["signals"]

    env = MockTradingEnvironment(initial_capital=10000.0, commission=0.001)
    trader = SignalTrader(env)
    portfolio_df = trader.execute_signals(prices, signals, symbol="AAPL")

    equity = portfolio_df["value"]

    buy_hold = 10000.0 * (prices / float(prices.iloc[0]))

    perf = PerformanceMetrics(equity_curve=equity, benchmark=buy_hold, risk_free_rate=0.0)
    perf.print_metrics(periods_per_year=252)

    print("saving equity curve")
    perf.plot_equity_curve(save_path="equity_curve.png")

    print("saving drawdown")
    perf.plot_drawdown(save_path="drawdown.png")

    print("done")




if __name__ == "__main__":
    _self_test()
