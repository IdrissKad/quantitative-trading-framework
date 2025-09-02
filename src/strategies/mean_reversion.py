import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .base import BaseStrategy, Signal, OrderSide


class MeanReversionStrategy(BaseStrategy):
    
    def __init__(
        self,
        name: str = "Mean Reversion",
        universe: List[str] = None,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        use_bollinger_bands: bool = True,
        use_rsi: bool = False,
        position_sizing: str = "risk_based",
        **kwargs
    ):
        super().__init__(
            name=name,
            universe=universe or [],
            lookback_period=lookback_period,
            position_sizing=position_sizing,
            **kwargs
        )
        
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_bollinger_bands = use_bollinger_bands
        self.use_rsi = use_rsi
        
    def on_initialize(self):
        self.z_scores = {}
        self.moving_averages = {}
        self.standard_deviations = {}
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        for symbol in self.universe:
            if symbol not in data.columns:
                continue
                
            symbol_data = data[symbol]
            
            if self.use_bollinger_bands:
                signal = self.generate_bollinger_signal(symbol, symbol_data)
            elif self.use_rsi:
                signal = self.generate_rsi_signal(symbol, symbol_data)
            else:
                signal = self.generate_zscore_signal(symbol, symbol_data)
                
            if signal:
                signals.append(signal)
                
        return signals
    
    def generate_bollinger_signal(
        self,
        symbol: str,
        price_data: pd.Series
    ) -> Optional[Signal]:
        
        if len(price_data) < self.lookback_period:
            return None
            
        moving_avg = price_data.rolling(window=self.lookback_period).mean()
        std_dev = price_data.rolling(window=self.lookback_period).std()
        
        upper_band = moving_avg + (self.entry_threshold * std_dev)
        lower_band = moving_avg - (self.entry_threshold * std_dev)
        
        current_price = price_data.iloc[-1]
        current_ma = moving_avg.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        z_score = (current_price - current_ma) / std_dev.iloc[-1] if std_dev.iloc[-1] > 0 else 0
        
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if abs(z_score) < self.exit_threshold:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=1.0,
                    confidence=0.7,
                    metadata={
                        "reason": "mean_reversion_exit",
                        "z_score": z_score
                    }
                )
        else:
            if current_price <= current_lower:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=min(1.0, abs(z_score) / 3),
                    confidence=min(0.9, abs(z_score) / 4),
                    metadata={
                        "entry_type": "lower_band",
                        "z_score": z_score,
                        "price": current_price,
                        "lower_band": current_lower
                    }
                )
            elif current_price >= current_upper:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=min(1.0, abs(z_score) / 3),
                    confidence=min(0.9, abs(z_score) / 4),
                    metadata={
                        "entry_type": "upper_band",
                        "z_score": z_score,
                        "price": current_price,
                        "upper_band": current_upper
                    }
                )
                
        return None
    
    def generate_rsi_signal(
        self,
        symbol: str,
        price_data: pd.Series
    ) -> Optional[Signal]:
        
        if len(price_data) < 14:
            return None
            
        rsi = self.calculate_rsi(price_data)
        current_rsi = rsi.iloc[-1]
        
        if symbol in self.positions:
            if 45 <= current_rsi <= 55:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=1.0,
                    confidence=0.6,
                    metadata={
                        "reason": "rsi_neutral",
                        "rsi": current_rsi
                    }
                )
        else:
            if current_rsi < 30:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=(30 - current_rsi) / 30,
                    confidence=min(0.8, (30 - current_rsi) / 20),
                    metadata={
                        "entry_type": "oversold",
                        "rsi": current_rsi
                    }
                )
            elif current_rsi > 70:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=(current_rsi - 70) / 30,
                    confidence=min(0.8, (current_rsi - 70) / 20),
                    metadata={
                        "entry_type": "overbought",
                        "rsi": current_rsi
                    }
                )
                
        return None
    
    def generate_zscore_signal(
        self,
        symbol: str,
        price_data: pd.Series
    ) -> Optional[Signal]:
        
        if len(price_data) < self.lookback_period:
            return None
            
        mean = price_data.rolling(window=self.lookback_period).mean().iloc[-1]
        std = price_data.rolling(window=self.lookback_period).std().iloc[-1]
        
        if std == 0:
            return None
            
        current_price = price_data.iloc[-1]
        z_score = (current_price - mean) / std
        
        self.z_scores[symbol] = z_score
        
        if symbol in self.positions:
            if abs(z_score) < self.exit_threshold:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=1.0,
                    confidence=0.7,
                    metadata={
                        "reason": "z_score_exit",
                        "z_score": z_score
                    }
                )
        else:
            if z_score < -self.entry_threshold:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=min(1.0, abs(z_score) / 3),
                    confidence=min(0.9, abs(z_score) / 4),
                    metadata={
                        "entry_type": "z_score_long",
                        "z_score": z_score
                    }
                )
            elif z_score > self.entry_threshold:
                return Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=min(1.0, abs(z_score) / 3),
                    confidence=min(0.9, abs(z_score) / 4),
                    metadata={
                        "entry_type": "z_score_short",
                        "z_score": z_score
                    }
                )
                
        return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        
        delta = prices.diff()
        
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: float
    ) -> float:
        
        if self.position_sizing == "risk_based":
            z_score = signal.metadata.get("z_score", 2.0)
            
            confidence = min(1.0, abs(z_score) / 4)
            
            base_size = portfolio_value * self.risk_per_trade
            
            position_size = base_size * confidence / current_price
            
            max_position = portfolio_value * 0.1 / current_price
            position_size = min(position_size, max_position)
            
            return position_size
            
        else:
            return super().calculate_position_size(signal, current_price, portfolio_value)


class StatisticalArbitrage(MeanReversionStrategy):
    
    def __init__(
        self,
        name: str = "Statistical Arbitrage",
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        use_kalman_filter: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            lookback_period=lookback_period,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            use_bollinger_bands=False,
            **kwargs
        )
        
        self.use_kalman_filter = use_kalman_filter
        self.spread_history = {}
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        if len(self.universe) < 2:
            return signals
            
        pairs = self.identify_cointegrated_pairs(data)
        
        for pair in pairs:
            signal = self.generate_pair_signal(pair, data)
            if signal:
                signals.extend(signal)
                
        return signals
    
    def identify_cointegrated_pairs(
        self,
        data: pd.DataFrame
    ) -> List[tuple]:
        
        pairs = []
        
        for i, symbol1 in enumerate(self.universe[:-1]):
            for symbol2 in self.universe[i+1:]:
                if symbol1 in data.columns and symbol2 in data.columns:
                    if self.test_cointegration(data[symbol1], data[symbol2]):
                        pairs.append((symbol1, symbol2))
                        
        return pairs
    
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> bool:
        
        correlation = series1.corr(series2)
        return abs(correlation) > 0.7
    
    def generate_pair_signal(
        self,
        pair: tuple,
        data: pd.DataFrame
    ) -> Optional[List[Signal]]:
        
        symbol1, symbol2 = pair
        
        if symbol1 not in data.columns or symbol2 not in data.columns:
            return None
            
        price1 = data[symbol1]
        price2 = data[symbol2]
        
        if len(price1) < self.lookback_period:
            return None
            
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        spread = price1 - hedge_ratio * price2
        
        spread_mean = spread.rolling(window=self.lookback_period).mean().iloc[-1]
        spread_std = spread.rolling(window=self.lookback_period).std().iloc[-1]
        
        if spread_std == 0:
            return None
            
        z_score = (spread.iloc[-1] - spread_mean) / spread_std
        
        pair_key = f"{symbol1}_{symbol2}"
        
        if pair_key in self.positions:
            if abs(z_score) < self.exit_threshold:
                return [
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol1,
                        side=OrderSide.SELL,
                        strength=1.0,
                        confidence=0.7,
                        metadata={"reason": "pair_exit", "z_score": z_score}
                    ),
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol2,
                        side=OrderSide.SELL,
                        strength=1.0,
                        confidence=0.7,
                        metadata={"reason": "pair_exit", "z_score": z_score}
                    )
                ]
        else:
            if z_score > self.entry_threshold:
                return [
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol1,
                        side=OrderSide.SELL,
                        strength=min(1.0, abs(z_score) / 3),
                        confidence=min(0.9, abs(z_score) / 4),
                        metadata={"pair": pair_key, "z_score": z_score}
                    ),
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol2,
                        side=OrderSide.BUY,
                        strength=min(1.0, abs(z_score) / 3) * hedge_ratio,
                        confidence=min(0.9, abs(z_score) / 4),
                        metadata={"pair": pair_key, "z_score": z_score}
                    )
                ]
            elif z_score < -self.entry_threshold:
                return [
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol1,
                        side=OrderSide.BUY,
                        strength=min(1.0, abs(z_score) / 3),
                        confidence=min(0.9, abs(z_score) / 4),
                        metadata={"pair": pair_key, "z_score": z_score}
                    ),
                    Signal(
                        timestamp=datetime.now(),
                        symbol=symbol2,
                        side=OrderSide.SELL,
                        strength=min(1.0, abs(z_score) / 3) * hedge_ratio,
                        confidence=min(0.9, abs(z_score) / 4),
                        metadata={"pair": pair_key, "z_score": z_score}
                    )
                ]
                
        return None
    
    def calculate_hedge_ratio(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> float:
        
        returns1 = series1.pct_change().dropna()
        returns2 = series2.pct_change().dropna()
        
        if len(returns1) < 2 or len(returns2) < 2:
            return 1.0
            
        covariance = returns1.cov(returns2)
        variance2 = returns2.var()
        
        if variance2 == 0:
            return 1.0
            
        return covariance / variance2