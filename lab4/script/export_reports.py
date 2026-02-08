import pandas as pd
from signal_generator import generate_hybrid_signals
from mock_trading import MockTradingEnvironment, SignalTrader
from performance_metrics import PerformanceMetrics
import sys
from data_collecting.py import DataHist


def load_prices(df) -> pd.Series:
    #df = pd.read_csv(f"data/{symbol}_10y_1d.csv")

    date_col = None
    for c in df.columns:
        if c.lower() in ["date", "datetime", "timestamp"]:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Date column not found. Columns: {list(df.columns)}")

    price_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["close", "adj close", "adj_close", "adjclose"]:
            price_col = c
            break
    if price_col is None:
        raise ValueError(f"Price column not found. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    prices = df[price_col].astype(float)
    prices.name = "close"
    return prices


def main():
    symbol = sys.argv[1]
    c = DataHist(symbol, 365*10, "Standard Indicators", interval="1d")
    data = c.request_data()
    prices = load_prices(data)

    initial_capital = 10000.0
    commission = 0.001

    result = generate_hybrid_signals(prices, mode="confirm")
    signals = result["signals"]

    env = MockTradingEnvironment(initial_capital=initial_capital, commission=commission)
    trader = SignalTrader(env)
    portfolio_df = trader.execute_signals(prices, signals, symbol="AAPL")

    equity = portfolio_df["value"]

    buy_hold = initial_capital * (prices / float(prices.iloc[0]))

    perf = PerformanceMetrics(equity_curve=equity, benchmark=buy_hold, risk_free_rate=0.0)
    metrics = perf.get_all_metrics(periods_per_year=252)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("metrics_summary.csv", index=False)

    portfolio_df.to_csv("portfolio_values.csv")

    env.export_trade_history_csv("trade_history.csv")

    perf.plot_equity_curve(save_path="equity_curve.png")
    perf.plot_drawdown(save_path="drawdown.png")

    print("saved: metrics_summary.csv")
    print("saved: portfolio_values.csv")
    print("saved: trade_history.csv")
    print("saved: equity_curve.png")
    print("saved: drawdown.png")


if __name__ == "__main__":
    main()
