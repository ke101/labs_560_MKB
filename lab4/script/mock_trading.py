from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional


class MockTradingEnvironment:
    """
    Mock trading account:
    - manage cash and positions
    - execute buy/sell with commission
    - keep trade history
    """

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.commission = float(commission)

        self.positions: Dict[str, int] = {}
        self.trade_history: List[dict] = []

    def buy(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        shares: Optional[int] = None,
        amount: Optional[float] = None,
    ) -> bool:
        if amount is not None:
            shares = int(amount // price)
        if shares is None or shares <= 0:
            return False

        cost = shares * price
        fee = cost * self.commission
        total_cost = cost + fee

        if total_cost > self.cash:
            return False

        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares

        self.trade_history.append(
            {
                "date": pd.to_datetime(date),
                "symbol": symbol,
                "action": "BUY",
                "shares": shares,
                "price": float(price),
                "gross": float(cost),
                "commission": float(fee),
                "net_cash_change": -float(total_cost),
                "cash_after": float(self.cash),
                "shares_after": int(self.positions[symbol]),
            }
        )
        return True

    def sell(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        shares: Optional[int] = None,
    ) -> bool:
        current = self.positions.get(symbol, 0)
        if current <= 0:
            return False

        if shares is None:
            shares = current
        shares = min(int(shares), current)
        if shares <= 0:
            return False

        revenue = shares * price
        fee = revenue * self.commission
        net_revenue = revenue - fee

        self.cash += net_revenue
        self.positions[symbol] = current - shares

        self.trade_history.append(
            {
                "date": pd.to_datetime(date),
                "symbol": symbol,
                "action": "SELL",
                "shares": shares,
                "price": float(price),
                "gross": float(revenue),
                "commission": float(fee),
                "net_cash_change": float(net_revenue),
                "cash_after": float(self.cash),
                "shares_after": int(self.positions[symbol]),
            }
        )
        return True

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        holdings_value = 0.0
        for sym, sh in self.positions.items():
            holdings_value += sh * float(prices.get(sym, 0.0))
        return float(self.cash + holdings_value)

    def get_summary(self) -> Dict:
        total_commission = 0.0
        for t in self.trade_history:
            total_commission += float(t.get("commission", 0.0))
        return {
            "initial_capital": float(self.initial_capital),
            "cash": float(self.cash),
            "positions": dict(self.positions),
            "num_trades": int(len(self.trade_history)),
            "total_commission": float(total_commission),
        }

    def get_trade_history_df(self) -> pd.DataFrame:
        if len(self.trade_history) == 0:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "action",
                    "shares",
                    "price",
                    "gross",
                    "commission",
                    "net_cash_change",
                    "cash_after",
                    "shares_after",
                ]
            )
        df = pd.DataFrame(self.trade_history)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    
    def export_trade_history_csv(self, filepath: str) -> None:
        df = self.get_trade_history_df()
        df.to_csv(filepath, index=False)


class SignalTrader:
    """
    Execute trades based on signal series:
    signal == 1: buy
    signal == -1: sell
    signal == 0: hold
    """

    def __init__(self, env: MockTradingEnvironment):
        self.env = env

    def execute_signals(
        self,
        prices: pd.Series,
        signals: pd.Series,
        symbol: str = "AAPL",
        buy_cash_fraction: float = 0.95,
    ) -> pd.DataFrame:
        prices = prices.copy()
        signals = signals.copy()

        prices.index = pd.to_datetime(prices.index)
        signals.index = pd.to_datetime(signals.index)

        # Align by intersection of dates
        idx = prices.index.intersection(signals.index)
        prices = prices.loc[idx]
        signals = signals.loc[idx]

        rows = []
        for date in idx:
            price = float(prices.loc[date])
            sig = int(signals.loc[date])

            if sig == 1:
                # use a fraction of current cash to buy
                amount = self.env.cash * float(buy_cash_fraction)
                self.env.buy(symbol, price, date, amount=amount)
            elif sig == -1:
                # sell all holdings
                self.env.sell(symbol, price, date, shares=None)

            shares = int(self.env.positions.get(symbol, 0))
            holdings_value = float(shares * price)
            value = float(self.env.cash + holdings_value)

            rows.append(
                {
                    "date": date,
                    "value": value,
                    "cash": float(self.env.cash),
                    "holdings_value": holdings_value,
                    "shares": shares,
                }
            )

        df = pd.DataFrame(rows).set_index("date")
        return df



def _self_test():
    import pandas as pd
    from signal_generator import generate_hybrid_signals

    csv_path = "AAPL_10y_1d.csv"
    df = pd.read_csv(csv_path)

    # detect columns
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

    result = generate_hybrid_signals(prices, mode="confirm")
    signals = result["signals"]

    env = MockTradingEnvironment(initial_capital=10000.0, commission=0.001)
    trader = SignalTrader(env)
    portfolio_df = trader.execute_signals(prices, signals, symbol="AAPL")

    print("portfolio_df head:")
    print(portfolio_df.head(5))
    print("")
    print("portfolio_df tail:")
    print(portfolio_df.tail(5))
    print("")
    print("final summary:")
    print(env.get_summary())
    print("")
    print("num trades in history:", len(env.get_trade_history_df()))

if __name__ == "__main__":
    _self_test()


