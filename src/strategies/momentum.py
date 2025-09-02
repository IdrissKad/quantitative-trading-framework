import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .base import BaseStrategy, Signal, OrderSide


class MomentumStrategy(BaseStrategy):
    
    def __init__(
        self,
        name: str = "Momentum",
        universe: List[str] = None,
        lookback_period: int = 20,
        holding_period: int = 5,
        rebalance_frequency: str = "daily",
        num_stocks: int = 10,
        momentum_metric: str = "return",
        position_sizing: str = "equal_weight",
        **kwargs
    ):
        super().__init__(
            name=name,
            universe=universe or [],
            lookback_period=lookback_period,
            rebalance_frequency=rebalance_frequency,
            position_sizing=position_sizing,
            max_positions=num_stocks,
            **kwargs
        )
        
        self.holding_period = holding_period
        self.num_stocks = num_stocks
        self.momentum_metric = momentum_metric
        self.rebalance_counter = 0
        self.rankings = {}
        
    def on_initialize(self):
        self.rebalance_counter = 0
        self.rankings = {}
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        self.rebalance_counter += 1
        
        if self.should_rebalance():
            momentum_scores = self.calculate_momentum(data)
            
            top_stocks = momentum_scores.nlargest(self.num_stocks)
            bottom_stocks = momentum_scores.nsmallest(self.num_stocks)
            
            current_holdings = set(self.positions.keys())
            new_holdings = set(top_stocks.index)
            
            to_sell = current_holdings - new_holdings
            for symbol in to_sell:
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    strength=1.0,
                    confidence=0.8,
                    metadata={"reason": "exit_momentum"}
                ))
            
            to_buy = new_holdings - current_holdings
            for symbol in to_buy:
                score = momentum_scores[symbol]
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    strength=self.normalize_score(score),
                    confidence=self.calculate_confidence(score, momentum_scores),
                    metadata={
                        "momentum_score": score,
                        "rank": list(top_stocks.index).index(symbol) + 1
                    }
                ))
                
        return signals
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        
        if self.momentum_metric == "return":
            momentum = (data.iloc[-1] / data.iloc[0]) - 1
            
        elif self.momentum_metric == "sharpe":
            returns = data.pct_change().dropna()
            momentum = returns.mean() / returns.std() * np.sqrt(252)
            
        elif self.momentum_metric == "relative_strength":
            gains = data.pct_change().clip(lower=0)
            losses = -data.pct_change().clip(upper=0)
            
            avg_gains = gains.rolling(window=14).mean()
            avg_losses = losses.rolling(window=14).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            momentum = rsi.iloc[-1]
            
        elif self.momentum_metric == "price_volume":
            if "volume" in data.columns:
                returns = data["close"].pct_change()
                volume_weighted = returns * data["volume"]
                momentum = volume_weighted.rolling(window=self.lookback_period).sum()
                momentum = momentum.iloc[-1]
            else:
                momentum = (data.iloc[-1] / data.iloc[0]) - 1
                
        else:
            momentum = (data.iloc[-1] / data.iloc[0]) - 1
            
        return momentum
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: float
    ) -> float:
        
        if self.position_sizing == "equal_weight":
            return portfolio_value / self.num_stocks / current_price
            
        elif self.position_sizing == "momentum_weighted":
            weight = signal.strength / self.num_stocks
            return portfolio_value * weight / current_price
            
        elif self.position_sizing == "risk_parity":
            volatility = self.calculate_volatility(signal.symbol)
            target_risk = 0.01
            position_size = (portfolio_value * target_risk) / (volatility * current_price)
            return position_size
            
        else:
            return portfolio_value / self.num_stocks / current_price
    
    def should_rebalance(self) -> bool:
        
        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return self.rebalance_counter % 5 == 0
        elif self.rebalance_frequency == "monthly":
            return self.rebalance_counter % 21 == 0
        else:
            return self.rebalance_counter % self.holding_period == 0
    
    def normalize_score(self, score: float) -> float:
        return min(1.0, max(0.0, (score + 1) / 2))
    
    def calculate_confidence(self, score: float, all_scores: pd.Series) -> float:
        percentile = (all_scores < score).sum() / len(all_scores)
        return percentile
    
    def calculate_volatility(self, symbol: str) -> float:
        return 0.2


class CrossSectionalMomentum(MomentumStrategy):
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Cross-Sectional Momentum",
            momentum_metric="return",
            **kwargs
        )
        
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        
        returns = (data.iloc[-1] / data.iloc[-self.lookback_period]) - 1
        
        skip_period = max(1, self.lookback_period // 10)
        adjusted_returns = (data.iloc[-skip_period] / data.iloc[-self.lookback_period]) - 1
        
        return adjusted_returns


class TimeSeriesMomentum(BaseStrategy):
    
    def __init__(
        self,
        name: str = "Time Series Momentum",
        universe: List[str] = None,
        lookback_periods: List[int] = None,
        position_sizing: str = "volatility_scaled",
        **kwargs
    ):
        super().__init__(
            name=name,
            universe=universe or [],
            **kwargs
        )
        
        self.lookback_periods = lookback_periods or [10, 20, 60, 120]
        self.position_sizing = position_sizing
        
    def on_initialize(self):
        self.volatilities = {}
        self.signals_history = []
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        for symbol in self.universe:
            if symbol not in data.columns:
                continue
                
            symbol_data = data[symbol]
            
            momentum_signals = []
            for period in self.lookback_periods:
                if len(symbol_data) >= period:
                    momentum = (symbol_data.iloc[-1] / symbol_data.iloc[-period]) - 1
                    momentum_signals.append(1 if momentum > 0 else -1)
                    
            if momentum_signals:
                avg_signal = np.mean(momentum_signals)
                
                if abs(avg_signal) > 0.5:
                    side = OrderSide.BUY if avg_signal > 0 else OrderSide.SELL
                    
                    if symbol in self.positions:
                        current_side = self.positions[symbol].side
                        if (side == OrderSide.BUY and current_side != "long") or \
                           (side == OrderSide.SELL and current_side != "short"):
                            signals.append(Signal(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                side=OrderSide.SELL,
                                strength=1.0,
                                confidence=0.7,
                                metadata={"reason": "close_position"}
                            ))
                    
                    signals.append(Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        side=side,
                        strength=abs(avg_signal),
                        confidence=min(0.9, abs(avg_signal)),
                        metadata={
                            "signal_strength": avg_signal,
                            "lookback_signals": momentum_signals
                        }
                    ))
                    
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: float
    ) -> float:
        
        if self.position_sizing == "volatility_scaled":
            volatility = self.calculate_asset_volatility(signal.symbol)
            target_volatility = 0.15
            
            position_size = (portfolio_value * target_volatility) / (volatility * current_price)
            
            position_size *= signal.strength
            
            max_position = portfolio_value * 0.1
            position_size = min(position_size, max_position / current_price)
            
            return position_size
            
        else:
            return super().calculate_position_size(signal, current_price, portfolio_value)
    
    def calculate_asset_volatility(self, symbol: str) -> float:
        if symbol in self.volatilities:
            return self.volatilities[symbol]
        return 0.2