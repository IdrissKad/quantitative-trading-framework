import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum

from ..strategies.base import BaseStrategy, Order, OrderType, OrderSide, PositionSide
from ..risk.manager import RiskManager
from ..analytics.metrics import PerformanceMetrics


class ExecutionModel(Enum):
    CLOSE = "close"
    NEXT_OPEN = "next_open"
    VWAP = "vwap"
    TWAP = "twap"
    LIMIT_ORDER = "limit_order"


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    min_commission: float = 1.0
    execution_model: ExecutionModel = ExecutionModel.CLOSE
    use_adjusted_close: bool = True
    rebalance_frequency: str = "daily"
    benchmark: Optional[str] = "SPY"
    risk_free_rate: float = 0.02
    margin_rate: float = 0.03
    short_rate: float = 0.01


@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    positions: pd.DataFrame
    returns: pd.Series
    metrics: Dict[str, float]
    signals: pd.DataFrame
    drawdown: pd.Series
    exposure: pd.Series
    benchmark_returns: Optional[pd.Series] = None


class BacktestEngine:
    
    def __init__(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig = BacktestConfig(),
        risk_manager: Optional[RiskManager] = None
    ):
        self.strategy = strategy
        self.config = config
        self.risk_manager = risk_manager or RiskManager()
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.positions_history: List[Dict] = []
        self.signals_history: List[Dict] = []
        
        self.current_cash = config.initial_capital
        self.current_positions: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
        
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        
        self.logger.info(f"Starting backtest for {self.strategy.name}")
        
        data = self._prepare_data(data, start_date, end_date)
        
        self.strategy.initialize(self.config.initial_capital, data.index[0])
        
        for timestamp, row in data.iterrows():
            self._process_bar(timestamp, row, data)
            
        result = self._generate_results(data)
        
        self.logger.info("Backtest completed")
        
        return result
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if self.config.use_adjusted_close and "adj_close" in data.columns:
            data["close"] = data["adj_close"]
            
        return data
    
    def _process_bar(
        self,
        timestamp: datetime,
        current_bar: pd.Series,
        full_data: pd.DataFrame
    ):
        
        self._update_positions(current_bar)
        
        lookback_data = self._get_lookback_data(
            full_data,
            timestamp,
            self.strategy.lookback_period
        )
        
        orders = self.strategy.on_data(timestamp, lookback_data)
        
        orders = self.risk_manager.filter_orders(
            orders,
            self.current_positions,
            self.current_cash,
            self.strategy.portfolio_value
        )
        
        for order in orders:
            self._execute_order(order, current_bar, timestamp)
            
        portfolio_value = self._calculate_portfolio_value(current_bar)
        self.equity_curve.append(portfolio_value)
        
        self._record_positions(timestamp)
        
    def _get_lookback_data(
        self,
        data: pd.DataFrame,
        current_time: datetime,
        lookback_period: int
    ) -> pd.DataFrame:
        
        current_idx = data.index.get_loc(current_time)
        start_idx = max(0, current_idx - lookback_period + 1)
        
        return data.iloc[start_idx:current_idx + 1]
    
    def _execute_order(
        self,
        order: Order,
        current_bar: pd.Series,
        timestamp: datetime
    ):
        
        execution_price = self._get_execution_price(order, current_bar)
        
        slippage_cost = execution_price * self.config.slippage * order.quantity
        
        commission = max(
            self.config.min_commission,
            execution_price * order.quantity * self.config.commission
        )
        
        total_cost = execution_price * order.quantity + commission + slippage_cost
        
        if order.side == OrderSide.BUY:
            if total_cost > self.current_cash:
                self.logger.warning(f"Insufficient cash for order: {order}")
                return
                
            self.current_cash -= total_cost
            
            if order.symbol in self.current_positions:
                position = self.current_positions[order.symbol]
                position["quantity"] += order.quantity
                position["avg_price"] = (
                    (position["avg_price"] * position["quantity"] + 
                     execution_price * order.quantity) /
                    (position["quantity"] + order.quantity)
                )
            else:
                self.current_positions[order.symbol] = {
                    "quantity": order.quantity,
                    "avg_price": execution_price,
                    "side": PositionSide.LONG,
                    "entry_time": timestamp
                }
                
        else:  # SELL
            if order.symbol not in self.current_positions:
                self.logger.warning(f"No position to sell: {order.symbol}")
                return
                
            position = self.current_positions[order.symbol]
            
            if position["quantity"] < order.quantity:
                self.logger.warning(f"Insufficient quantity to sell: {order}")
                return
                
            pnl = (execution_price - position["avg_price"]) * order.quantity
            
            self.current_cash += execution_price * order.quantity - commission - slippage_cost
            
            position["quantity"] -= order.quantity
            
            if position["quantity"] == 0:
                del self.current_positions[order.symbol]
                
            trade = Trade(
                timestamp=timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                commission=commission,
                slippage=slippage_cost,
                pnl=pnl
            )
            
            self.trades.append(trade)
            
        self.strategy.add_position(
            order.symbol,
            PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT,
            order.quantity,
            execution_price,
            timestamp
        )
    
    def _get_execution_price(self, order: Order, current_bar: pd.Series) -> float:
        
        if self.config.execution_model == ExecutionModel.CLOSE:
            base_price = current_bar.get("close", current_bar.get(order.symbol, 0))
        elif self.config.execution_model == ExecutionModel.NEXT_OPEN:
            base_price = current_bar.get("open", current_bar.get(order.symbol, 0))
        elif self.config.execution_model == ExecutionModel.VWAP:
            base_price = (
                current_bar.get("high", 0) + 
                current_bar.get("low", 0) + 
                current_bar.get("close", 0)
            ) / 3
        else:
            base_price = current_bar.get("close", current_bar.get(order.symbol, 0))
            
        if order.side == OrderSide.BUY:
            return base_price * (1 + self.config.slippage)
        else:
            return base_price * (1 - self.config.slippage)
    
    def _update_positions(self, current_bar: pd.Series):
        
        for symbol, position in self.current_positions.items():
            if symbol in current_bar.index:
                current_price = current_bar[symbol]
            elif "close" in current_bar.index:
                current_price = current_bar["close"]
            else:
                continue
                
            position["current_price"] = current_price
            position["unrealized_pnl"] = (
                (current_price - position["avg_price"]) * position["quantity"]
            )
    
    def _calculate_portfolio_value(self, current_bar: pd.Series) -> float:
        
        positions_value = 0
        for symbol, position in self.current_positions.items():
            if symbol in current_bar.index:
                current_price = current_bar[symbol]
            elif "close" in current_bar.index:
                current_price = current_bar["close"]
            else:
                current_price = position.get("current_price", position["avg_price"])
                
            positions_value += current_price * position["quantity"]
            
        return self.current_cash + positions_value
    
    def _record_positions(self, timestamp: datetime):
        
        positions_snapshot = {
            "timestamp": timestamp,
            "cash": self.current_cash,
            "positions": self.current_positions.copy()
        }
        self.positions_history.append(positions_snapshot)
    
    def _generate_results(self, data: pd.DataFrame) -> BacktestResult:
        
        equity_series = pd.Series(
            self.equity_curve,
            index=data.index[:len(self.equity_curve)]
        )
        
        returns = equity_series.pct_change().fillna(0)
        
        drawdown = self._calculate_drawdown(equity_series)
        
        exposure = self._calculate_exposure()
        
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades]) if self.trades else pd.DataFrame()
        
        positions_df = pd.DataFrame(self.positions_history)
        
        signals_df = pd.DataFrame([s.__dict__ for s in self.strategy.signals]) if self.strategy.signals else pd.DataFrame()
        
        metrics_calculator = PerformanceMetrics()
        metrics = metrics_calculator.calculate_all_metrics(
            returns=returns,
            equity_curve=equity_series,
            trades=trades_df,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate
        )
        
        benchmark_returns = None
        if self.config.benchmark:
            benchmark_returns = self._get_benchmark_returns(data.index)
            
        return BacktestResult(
            equity_curve=equity_series,
            trades=self.trades,
            positions=positions_df,
            returns=returns,
            metrics=metrics,
            signals=signals_df,
            drawdown=drawdown,
            exposure=exposure,
            benchmark_returns=benchmark_returns
        )
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    def _calculate_exposure(self) -> pd.Series:
        
        exposure_values = []
        for snapshot in self.positions_history:
            total_exposure = sum(
                pos["quantity"] * pos.get("current_price", pos["avg_price"])
                for pos in snapshot["positions"].values()
            )
            portfolio_value = snapshot["cash"] + total_exposure
            exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0
            exposure_values.append(exposure_pct)
            
        return pd.Series(exposure_values)
    
    def _get_benchmark_returns(self, index: pd.DatetimeIndex) -> Optional[pd.Series]:
        
        return None