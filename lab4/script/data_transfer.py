import pandas as pd
import pandas_ta as ta


class Category:
    def __init__(self, dataframe: pd.DataFrame, date_col: str = "datetime"):
        self.dataframe = dataframe
        self.date_col = date_col

    def createDataset(self):
        self.RA_5()  # RA_5
        self.RA_10()  # RA_10
        self.MACD()  # MACD
        self.CCI()  # CCI
        self.ATR()  # ATR
        self.BOLL()  # BollingerUpper, BollingerLower
        self.MA_5()  # MA_5
        self.MA_10()  # MA_10
        self.MTM_30()  # Momentum_30
        self.MTM_90()  # Momentum_90
        self.ROC()  # ROC
        self.SOD()  # SOD_14
        self.RSI()  # RSI_14
        self.EMA_12()  # EMA_12
        self.EMA_26()  # EMA_26
        self.OBV()  # OBV
        self.StochRSI()  # StochRSIk_14, StochRSId_14
        self.ADX()  # ADX_14
        self.CMF()  # CMF
        self.Vortex()  # Vortex_14
        self.TRIX()  # TRIX_30
        self.WILLR()  # WILLR
        self.KC()  # KCL_20, KCB_20, KCU_20
        self.DCL()  # DCL_20, DCM_20, DCU_20
        self.KAMA()  # KAMA
        self.PSAR()  # PSARl_0.02_0.2, "PSARs_0.02_0.2, PSARaf_0.02_0.2, PSARr_0.02_0.2
        return self.dataframe

    def RA_5(self):
        self.dataframe["RA_5"] = self.dataframe["Close"].rolling(window=5).std()

    def RA_10(self):
        self.dataframe["RA_10"] = self.dataframe["Close"].rolling(window=10).std()

    def MACD(self):
        self.dataframe["MACD"] = ta.macd(self.dataframe["Close"])["MACD_12_26_9"]

    def CCI(self):
        self.dataframe["CCI_20"] = ta.cci(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            length=20,
        )

    def ATR(self):
        self.dataframe["ATR"] = ta.atr(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            length=14,
        )

    def BOLL(self):
        bollinger = ta.bbands(self.dataframe["Close"])
        self.dataframe["BollingerUpper"] = bollinger["BBU_5_2.0"]
        self.dataframe["BollingerLower"] = bollinger["BBL_5_2.0"]

    def MA_5(self):
        self.dataframe["MA_5"] = ta.sma(self.dataframe["Close"], length=5)

    def MA_10(self):
        self.dataframe["MA_10"] = ta.sma(self.dataframe["Close"], length=10)

    def MTM_30(self):
        self.dataframe["Momentum_30"] = ta.mom(self.dataframe["Close"], length=30)

    def MTM_90(self):
        self.dataframe["Momentum_90"] = ta.mom(self.dataframe["Close"], length=90)

    def ROC(self):
        self.dataframe["ROC"] = ta.roc(self.dataframe["Close"], length=10)

    def SOD(self):
        self.dataframe["SOD_14"] = ta.stoch(
            self.dataframe["High"], self.dataframe["Low"], self.dataframe["Close"]
        )["STOCHk_14_3_3"]

    def RSI(self):
        self.dataframe["RSI_14"] = ta.rsi(self.dataframe["Close"], length=14)

    def EMA_12(self):
        self.dataframe["EMA_12"] = ta.ema(self.dataframe["Close"], length=12)

    def EMA_26(self):
        self.dataframe["EMA_26"] = ta.ema(self.dataframe["Close"], length=26)

    def OBV(self):
        self.dataframe["OBV"] = ta.obv(
            self.dataframe["Close"], self.dataframe["Volume"]
        )

    def StochRSI(self):
        stochrsi = ta.stochrsi(self.dataframe["Close"], length=14)
        self.dataframe["StochRSIk_14"] = stochrsi["STOCHRSIk_14_14_3_3"]
        self.dataframe["StochRSId_14"] = stochrsi["STOCHRSId_14_14_3_3"]

    def ADX(self):
        adx = ta.adx(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            length=14,
        )
        self.dataframe["ADX_14"] = adx["ADX_14"]

    def CMF(self):
        self.dataframe["CMF"] = ta.cmf(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            self.dataframe["Volume"],
        )

    def Vortex(self):
        vortex = ta.vortex(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            length=14,
        )
        self.dataframe["Vortex_14"] = vortex["VTXP_14"]

    def TRIX(self):
        trix = ta.trix(self.dataframe["Close"], length=30)
        self.dataframe["TRIX_30"] = trix["TRIX_30_9"]

    def WILLR(self):
        self.dataframe["WILLR"] = ta.willr(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            length=14,
        )

    def KC(self):
        kc = ta.kc(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
            scalar=2,
            length=20,
        )
        self.dataframe["KCL_20"] = kc["KCLe_20_2.0"]
        self.dataframe["KCB_20"] = kc["KCBe_20_2.0"]
        self.dataframe["KCU_20"] = kc["KCUe_20_2.0"]

    def DCL(self):
        dcl = ta.donchian(
            close=self.dataframe["Close"],
            lower_length=20,
            upper_length=20,
            high=self.dataframe["High"],
            low=self.dataframe["Low"],
        )
        self.dataframe["DCL_20"] = dcl["DCL_20_20"]
        self.dataframe["DCM_20"] = dcl["DCM_20_20"]
        self.dataframe["DCU_20"] = dcl["DCU_20_20"]

    def KAMA(self):
        self.dataframe["KAMA"] = ta.kama(
            self.dataframe["Close"], length=10, fast=2, slow=30
        )

    def PSAR(self):
        psar = ta.psar(
            self.dataframe["High"],
            self.dataframe["Low"],
            self.dataframe["Close"],
        )
        self.dataframe["PSARl"] = psar["PSARl_0.02_0.2"]
        self.dataframe["PSARs"] = psar["PSARs_0.02_0.2"]
        self.dataframe["PSARaf"] = psar["PSARaf_0.02_0.2"]
        self.dataframe["PSARr"] = psar["PSARr_0.02_0.2"]