# yahoo finance
from datetime import datetime, timedelta
import json
import os
from curl_cffi import requests as cffi_requests
import yfinance as yf
import pandas as pd

from data_transfer import Category

_YF_SESSION = cffi_requests.Session(impersonate="chrome")
DATE = "datetime"
AllowedIntervals = {
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
}
DayIntervals = {"1d", "5d", "1wk", "1mo", "3mo"}


class DataHist:
    def __init__(
        self, symbol: str, dataSize: int, indicators: str, interval="1d", end_date=None
    ):
        self.symbol = str(symbol)
        self.dataSize = int(dataSize)
        self.endDate = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        self.interval = interval
        assert self.interval in AllowedIntervals, f"Invalid interval {interval}!"

        self.features = self._get_features(indicators)
        self.cache_dir = "data_cache"
        self.cache_file = os.path.join(self.cache_dir, f"{symbol}_{interval}.csv")
        self.use_cache = False

    def _get_features(self, indicators: str):
        with open("indicators_config.json", "r") as f:
            indicators_set = json.load(f)
        return indicators_set[indicators]

    def _get_raw_data(self):
        if self._is_cache_valid():
            print("Using cached data...")
            return self._load_from_cache()

        print("Downloading new data...")
        new_data = self._download_new_data()
        return new_data

    def _process_dataframe(self, df: pd.DataFrame):
        if "Date" in df.columns:
            df = df.rename({"Date": DATE}, axis=1)
        elif "Datetime" in df.columns:
            df = df.rename({"Datetime": DATE}, axis=1)
            # convert to local time
            localTz = datetime.now().astimezone().tzinfo
            df[DATE] = df[DATE].dt.tz_convert(localTz)
        # to tz-naive
        df[DATE] = df[DATE].dt.tz_localize(None)
        df["Volume"] = df["Volume"].astype(float)
        df = df.round(4)
        df = df.iloc[::-1]
        return df

    def _save_to_cache(self, df: pd.DataFrame):
        if self.use_cache:
            cached_df = self._load_from_cache()
            combined_df = (
                pd.concat([cached_df, df])
                .drop_duplicates(subset=[DATE])
                .sort_values(by=DATE)
            )
        else:
            combined_df = df

        if not os.path.exists(self.cache_file) or not combined_df.equals(
            pd.read_csv(self.cache_file)
        ):
            os.makedirs(self.cache_dir, exist_ok=True)
            combined_df.to_csv(self.cache_file, index=False)

    def _load_from_cache(self):
        df = pd.read_csv(self.cache_file)
        df[DATE] = pd.to_datetime(df[DATE])
        return df

    def _is_cache_valid(self):
        if not os.path.exists(self.cache_file):
            return False

        cached_df = pd.read_csv(self.cache_file)
        if cached_df.empty:
            return False

        cached_df[DATE] = pd.to_datetime(cached_df[DATE])
        cached_df[DATE] = cached_df[DATE].dt.tz_localize(None)
        latest_cached_time = cached_df[DATE].max()

        # # Define how recent the last cached date should be for minute-level intervals
        # if self.interval in {"1m", "2m", "5m", "15m", "30m", "60m"}:
        #     recent_threshold = timedelta(minutes=15)
        # else:
        #     recent_threshold = timedelta(hours=1)

        # Get the latest trading data for comparison
        latest_data = yf.download(
            self.symbol, period="1d", interval=self.interval, progress=False, session=_YF_SESSION
        )
        latest_trading_time = latest_data.index.max()

        # Check if latest_trading_day has tz info and convert to local time if needed
        if latest_trading_time.tzinfo is not None:
            localTz = datetime.now().astimezone().tzinfo
            latest_trading_time = latest_trading_time.tz_convert(localTz).tz_localize(
                None
            )
        else:
            latest_trading_time = latest_trading_time.tz_localize(None)

        # Check if the cached data is recent enough
        self.use_cache = latest_cached_time >= latest_trading_time
        return self.use_cache

    def _download_new_data(self):
        if self.interval not in DayIntervals:
            start_date = (datetime.now() - timedelta(days=58)).strftime("%Y-%m-%d")
        else:
            start_date = None
        df_new: pd.DataFrame = (
            yf.download(
                self.symbol,
                start=start_date,
                end=self.endDate,
                interval=self.interval,
                ignore_tz=False,
                progress=False,
                session=_YF_SESSION
            )
            .reset_index()
            .iloc[::-1]
        )
        df_new.columns = df_new.columns.droplevel(1)
        df_new = self._process_dataframe(df_new)
        return df_new

    def _compute_indicators(self, df: pd.DataFrame, date_col: str):
        c = Category(df, date_col)
        df = c.createDataset()
        return df

    def _process_features(self, df: pd.DataFrame, required_columns: list[str]):
        df.set_index(DATE, inplace=True)
        for col in required_columns:
            if col not in df.columns:
                df[col] = float("nan")
                print(f"Missing column: {col}")

        df = df.ffill().bfill()
        return df[required_columns]

    def request_data(self, prediction_date=None, time_steps=None) -> pd.DataFrame:
        """
        Retrieves the data from yahoo finance and computes the indicators.

        Args:
            prediction_date (str, optional): The date for which the data is requested. If provided, the function will
                return a subset of the data that includes the given date. Defaults to None.
            time_steps (int, optional): The number of time steps to include in the subset of data. Only applicable if
                `prediction_date` is provided. Defaults to None.

        Returns:
            pd.DataFrame: The requested data as a pandas DataFrame.
        """
        df = self._get_raw_data()
        df = self._compute_indicators(df, DATE)
        self._save_to_cache(df)

        if prediction_date and time_steps:
            prediction_date = pd.to_datetime(prediction_date)
            available_dates = df[df[DATE] < prediction_date]
            if not available_dates.empty:
                pred_index = available_dates.index[-1]
            else:
                raise ValueError("No data available before the given prediction date")

            start_index = max(0, pred_index - time_steps + 1)
            df = df[start_index : pred_index + 1]

        if df.shape[0] >= self.dataSize:
            df = df[-self.dataSize :]

        df = self._process_features(df, self.features)

        return df


if __name__ == "__main__":
    c = DataHist("BTC-USD", 100, "Standard Indicators", interval="5m")
    print(c.request_data())