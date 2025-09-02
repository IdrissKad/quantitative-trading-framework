import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import yfinance as yf
from abc import ABC, abstractmethod
import asyncio
import aiohttp


class DataSource(Enum):
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"
    BINANCE = "binance"
    COINBASE = "coinbase"


class DataFrequency(Enum):
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DataProvider(ABC):
    
    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency
    ) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def fetch_realtime(
        self,
        symbol: str
    ) -> MarketData:
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        pass


class YahooDataProvider(DataProvider):
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        
        try:
            ticker = yf.Ticker(symbol)
            
            interval_map = {
                DataFrequency.MINUTE: "1m",
                DataFrequency.FIVE_MINUTE: "5m",
                DataFrequency.FIFTEEN_MINUTE: "15m",
                DataFrequency.THIRTY_MINUTE: "30m",
                DataFrequency.HOUR: "1h",
                DataFrequency.DAILY: "1d",
                DataFrequency.WEEKLY: "1wk",
                DataFrequency.MONTHLY: "1mo"
            }
            
            interval = interval_map.get(frequency, "1d")
            
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str) -> MarketData:
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=info.get("regularMarketOpen", 0),
                high=info.get("regularMarketDayHigh", 0),
                low=info.get("regularMarketDayLow", 0),
                close=info.get("regularMarketPrice", 0),
                volume=info.get("regularMarketVolume", 0),
                bid=info.get("bid", None),
                ask=info.get("ask", None),
                bid_size=info.get("bidSize", None),
                ask_size=info.get("askSize", None)
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching realtime data for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return "regularMarketPrice" in info
        except:
            return False


class DataPipeline:
    
    def __init__(
        self,
        provider: DataProvider = None,
        cache_enabled: bool = True,
        cache_dir: str = "./cache"
    ):
        self.provider = provider or YahooDataProvider()
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.preprocessors: List[Callable] = []
        self.validators: List[Callable] = []
        
    def add_preprocessor(self, func: Callable):
        self.preprocessors.append(func)
        
    def add_validator(self, func: Callable):
        self.validators.append(func)
        
    async def get_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        
        all_data = {}
        
        tasks = []
        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}_{frequency}"
            
            if self.cache_enabled and cache_key in self.data_cache:
                all_data[symbol] = self.data_cache[cache_key]
            else:
                tasks.append(self._fetch_symbol_data(
                    symbol, start_date, end_date, frequency
                ))
                
        if tasks:
            results = await asyncio.gather(*tasks)
            for symbol, data in zip(symbols, results):
                if not data.empty:
                    all_data[symbol] = data
                    
                    if self.cache_enabled:
                        cache_key = f"{symbol}_{start_date}_{end_date}_{frequency}"
                        self.data_cache[cache_key] = data
                        
        combined_data = self._combine_data(all_data)
        
        for preprocessor in self.preprocessors:
            combined_data = preprocessor(combined_data)
            
        if not self._validate_data(combined_data):
            self.logger.warning("Data validation failed")
            
        return combined_data
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency
    ) -> pd.DataFrame:
        
        try:
            data = await self.provider.fetch_historical(
                symbol, start_date, end_date, frequency
            )
            return data
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        
        if not data_dict:
            return pd.DataFrame()
            
        if len(data_dict) == 1:
            return list(data_dict.values())[0]
            
        close_prices = pd.DataFrame()
        
        for symbol, data in data_dict.items():
            if "close" in data.columns:
                close_prices[symbol] = data["close"]
                
        return close_prices
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        
        if data.empty:
            return False
            
        for validator in self.validators:
            if not validator(data):
                return False
                
        if data.isnull().sum().sum() > len(data) * 0.1:
            self.logger.warning("Too many missing values in data")
            return False
            
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        data = data.fillna(method="ffill").fillna(method="bfill")
        
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.01)
            Q99 = data[col].quantile(0.99)
            data[col] = data[col].clip(lower=Q1, upper=Q99)
            
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        
        for col in data.columns:
            if "close" in col.lower() or data[col].dtype in [np.float64, np.float32]:
                data[f"{col}_sma_20"] = data[col].rolling(window=20).mean()
                data[f"{col}_sma_50"] = data[col].rolling(window=50).mean()
                
                data[f"{col}_ema_12"] = data[col].ewm(span=12, adjust=False).mean()
                data[f"{col}_ema_26"] = data[col].ewm(span=26, adjust=False).mean()
                
                delta = data[col].diff()
                gains = delta.clip(lower=0)
                losses = -delta.clip(upper=0)
                avg_gains = gains.rolling(window=14).mean()
                avg_losses = losses.rolling(window=14).mean()
                rs = avg_gains / avg_losses
                data[f"{col}_rsi"] = 100 - (100 / (1 + rs))
                
                bb_sma = data[col].rolling(window=20).mean()
                bb_std = data[col].rolling(window=20).std()
                data[f"{col}_bb_upper"] = bb_sma + (2 * bb_std)
                data[f"{col}_bb_lower"] = bb_sma - (2 * bb_std)
                
        return data
    
    def resample_data(
        self,
        data: pd.DataFrame,
        target_frequency: str
    ) -> pd.DataFrame:
        
        resampled = data.resample(target_frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        
        return resampled
    
    def calculate_returns(
        self,
        data: pd.DataFrame,
        method: str = "simple"
    ) -> pd.DataFrame:
        
        if method == "simple":
            returns = data.pct_change()
        elif method == "log":
            returns = np.log(data / data.shift(1))
        else:
            returns = data.pct_change()
            
        return returns
    
    def clear_cache(self):
        self.data_cache.clear()
        self.logger.info("Data cache cleared")


class RealtimeDataStream:
    
    def __init__(
        self,
        symbols: List[str],
        callback: Callable,
        provider: DataProvider = None
    ):
        self.symbols = symbols
        self.callback = callback
        self.provider = provider or YahooDataProvider()
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        
        self.is_running = True
        self.logger.info(f"Starting realtime stream for {self.symbols}")
        
        while self.is_running:
            try:
                tasks = [
                    self.provider.fetch_realtime(symbol)
                    for symbol in self.symbols
                ]
                
                results = await asyncio.gather(*tasks)
                
                for data in results:
                    if data:
                        await self.callback(data)
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in realtime stream: {e}")
                await asyncio.sleep(5)
                
    def stop(self):
        self.is_running = False
        self.logger.info("Stopping realtime stream")