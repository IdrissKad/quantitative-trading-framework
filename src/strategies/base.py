from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    side: OrderSide
    strength: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float
    realized_pnl: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    
    def __init__(
        self,
        name: str,
        universe: List[str],
        lookback_period: int = 20,
        rebalance_frequency: str = "daily",
        position_sizing: str = "equal_weight",
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        **kwargs
    ):
        self.name = name
        self.universe = universe
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.position_sizing = position_sizing
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.params = kwargs
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.signals: List[Signal] = []
        self.portfolio_value: float = 0.0
        self.cash: float = 0.0
        self.is_initialized = False
        
    def initialize(self, initial_capital: float, start_date: datetime):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.start_date = start_date
        self.is_initialized = True
        self.on_initialize()
        
    @abstractmethod
    def on_initialize(self):
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal: Signal, 
        current_price: float,
        portfolio_value: float
    ) -> float:
        pass
    
    def on_data(self, timestamp: datetime, data: pd.DataFrame) -> List[Order]:
        if not self.is_initialized:
            raise RuntimeError("Strategy not initialized")
            
        self.update_positions(data)
        
        signals = self.generate_signals(data)
        self.signals.extend(signals)
        
        orders = self.signals_to_orders(signals, data)
        self.orders.extend(orders)
        
        return orders
    
    def signals_to_orders(
        self, 
        signals: List[Signal], 
        data: pd.DataFrame
    ) -> List[Order]:
        orders = []
        
        for signal in signals:
            if self.should_trade(signal):
                current_price = self.get_current_price(signal.symbol, data)
                position_size = self.calculate_position_size(
                    signal, 
                    current_price,
                    self.portfolio_value
                )
                
                if position_size > 0:
                    order = self.create_order(
                        signal.symbol,
                        signal.side,
                        position_size,
                        current_price
                    )
                    orders.append(order)
                    
        return orders
    
    def should_trade(self, signal: Signal) -> bool:
        if len(self.positions) >= self.max_positions:
            return False
            
        if signal.symbol in self.positions:
            return False
            
        if signal.confidence < 0.6:
            return False
            
        return True
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price if order_type != OrderType.MARKET else None
        )
    
    def update_positions(self, data: pd.DataFrame):
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, data)
            position.current_price = current_price
            
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (
                    (current_price - position.entry_price) * position.quantity
                )
            else:
                position.unrealized_pnl = (
                    (position.entry_price - current_price) * position.quantity
                )
    
    def get_current_price(self, symbol: str, data: pd.DataFrame) -> float:
        if symbol in data.columns:
            return data[symbol].iloc[-1]
        elif "close" in data.columns and len(data) > 0:
            return data["close"].iloc[-1]
        else:
            raise ValueError(f"Cannot get price for {symbol}")
    
    def add_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        timestamp: datetime
    ):
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            entry_time=timestamp,
            unrealized_pnl=0.0
        )
        
        cost = quantity * entry_price
        self.cash -= cost
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        if symbol not in self.positions:
            return 0.0
            
        position = self.positions[symbol]
        
        if position.side == PositionSide.LONG:
            realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
            
        position.realized_pnl = realized_pnl
        
        proceeds = position.quantity * exit_price
        self.cash += proceeds
        
        del self.positions[symbol]
        
        return realized_pnl
    
    def get_portfolio_value(self, data: pd.DataFrame) -> float:
        positions_value = sum(
            pos.quantity * self.get_current_price(pos.symbol, data)
            for pos in self.positions.values()
        )
        self.portfolio_value = self.cash + positions_value
        return self.portfolio_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        total_trades = len(self.orders)
        winning_trades = sum(1 for s in self.signals if s.strength > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_return": total_return,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "num_positions": len(self.positions),
            "total_trades": total_trades,
            "win_rate": win_rate
        }
    
    def reset(self):
        self.positions.clear()
        self.orders.clear()
        self.signals.clear()
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.is_initialized = False