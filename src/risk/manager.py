import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats


class RiskMetric(Enum):
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    BETA = "beta"
    CORRELATION = "correlation"


@dataclass
class RiskLimits:
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.02
    max_drawdown: float = 0.2
    max_leverage: float = 1.0
    max_concentration: float = 0.3
    max_correlation: float = 0.7
    var_limit: float = 0.05
    stop_loss: float = 0.02
    take_profit: float = 0.05
    max_daily_loss: float = 0.03


@dataclass
class RiskMetrics:
    portfolio_var: float
    portfolio_cvar: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    correlation_matrix: Optional[pd.DataFrame] = None
    exposure: float = 0.0
    leverage: float = 0.0
    concentration: Dict[str, float] = None


class PositionSizer:
    
    @staticmethod
    def kelly_criterion(
        win_probability: float,
        win_loss_ratio: float,
        kelly_fraction: float = 0.25
    ) -> float:
        
        if win_loss_ratio <= 0:
            return 0.0
            
        q = 1 - win_probability
        f = (win_probability * win_loss_ratio - q) / win_loss_ratio
        
        return max(0, min(1, f * kelly_fraction))
    
    @staticmethod
    def fixed_fractional(
        portfolio_value: float,
        risk_per_trade: float,
        stop_loss_distance: float
    ) -> float:
        
        if stop_loss_distance <= 0:
            return 0.0
            
        position_size = (portfolio_value * risk_per_trade) / stop_loss_distance
        
        return position_size
    
    @staticmethod
    def volatility_based(
        portfolio_value: float,
        target_volatility: float,
        asset_volatility: float,
        correlation: float = 0.0
    ) -> float:
        
        if asset_volatility <= 0:
            return 0.0
            
        position_size = (portfolio_value * target_volatility) / asset_volatility
        
        if correlation > 0:
            position_size *= (1 - correlation * 0.5)
            
        return position_size
    
    @staticmethod
    def equal_weight(
        portfolio_value: float,
        num_positions: int
    ) -> float:
        
        if num_positions <= 0:
            return 0.0
            
        return portfolio_value / num_positions
    
    @staticmethod
    def risk_parity(
        portfolio_value: float,
        asset_volatilities: List[float],
        correlations: Optional[np.ndarray] = None
    ) -> List[float]:
        
        n_assets = len(asset_volatilities)
        
        if correlations is None:
            correlations = np.eye(n_assets)
            
        inverse_vols = 1 / np.array(asset_volatilities)
        weights = inverse_vols / np.sum(inverse_vols)
        
        positions = [portfolio_value * w for w in weights]
        
        return positions


class RiskManager:
    
    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(__name__)
        self.position_sizer = PositionSizer()
        
        self.portfolio_history = []
        self.daily_returns = []
        self.positions_history = []
        
    def filter_orders(
        self,
        orders: List,
        current_positions: Dict,
        available_cash: float,
        portfolio_value: float
    ) -> List:
        
        filtered_orders = []
        
        for order in orders:
            if self._validate_order(order, current_positions, available_cash, portfolio_value):
                filtered_orders.append(order)
            else:
                self.logger.warning(f"Order rejected by risk manager: {order}")
                
        return filtered_orders
    
    def _validate_order(
        self,
        order,
        current_positions: Dict,
        available_cash: float,
        portfolio_value: float
    ) -> bool:
        
        order_value = order.quantity * order.price if order.price else 0
        
        if order_value > portfolio_value * self.risk_limits.max_position_size:
            self.logger.warning(f"Order exceeds max position size: {order_value / portfolio_value:.2%}")
            return False
            
        total_exposure = sum(
            pos.get("quantity", 0) * pos.get("current_price", pos.get("avg_price", 0))
            for pos in current_positions.values()
        ) + order_value
        
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if leverage > self.risk_limits.max_leverage:
            self.logger.warning(f"Order would exceed max leverage: {leverage:.2f}")
            return False
            
        if order.symbol in current_positions:
            existing_position = current_positions[order.symbol]
            new_position_value = (
                existing_position.get("quantity", 0) * 
                existing_position.get("current_price", existing_position.get("avg_price", 0)) +
                order_value
            )
            
            concentration = new_position_value / portfolio_value if portfolio_value > 0 else 0
            
            if concentration > self.risk_limits.max_concentration:
                self.logger.warning(f"Order would exceed max concentration: {concentration:.2%}")
                return False
                
        return True
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        if method == "historical":
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == "parametric":
            mean = returns.mean()
            std = returns.std()
            var = mean - std * stats.norm.ppf(confidence_level)
        elif method == "monte_carlo":
            simulated_returns = np.random.normal(
                returns.mean(),
                returns.std(),
                10000
            )
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        else:
            var = 0.0
            
        return abs(var)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= -var].mean()
        
        return abs(cvar) if not pd.isna(cvar) else var
    
    def calculate_portfolio_metrics(
        self,
        returns: pd.Series,
        positions: Dict[str, Any],
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        
        portfolio_var = self.calculate_var(returns)
        portfolio_cvar = self.calculate_cvar(returns)
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() > 0 else 0
        )
        
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        )
        
        total_exposure = sum(
            pos.get("quantity", 0) * pos.get("current_price", pos.get("avg_price", 0))
            for pos in positions.values()
        )
        
        portfolio_value = total_exposure + positions.get("cash", 0)
        exposure = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        concentration = {}
        if portfolio_value > 0:
            for symbol, pos in positions.items():
                if symbol != "cash":
                    pos_value = pos.get("quantity", 0) * pos.get("current_price", pos.get("avg_price", 0))
                    concentration[symbol] = pos_value / portfolio_value
                    
        return RiskMetrics(
            portfolio_var=portfolio_var,
            portfolio_cvar=portfolio_cvar,
            current_drawdown=current_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            exposure=exposure,
            concentration=concentration
        )
    
    def calculate_position_size(
        self,
        method: str,
        portfolio_value: float,
        signal_strength: float,
        volatility: float,
        **kwargs
    ) -> float:
        
        if method == "kelly":
            win_probability = kwargs.get("win_probability", 0.55)
            win_loss_ratio = kwargs.get("win_loss_ratio", 1.5)
            kelly_fraction = kwargs.get("kelly_fraction", 0.25)
            
            kelly_size = self.position_sizer.kelly_criterion(
                win_probability,
                win_loss_ratio,
                kelly_fraction
            )
            
            return portfolio_value * kelly_size * signal_strength
            
        elif method == "fixed_fractional":
            risk_per_trade = kwargs.get("risk_per_trade", self.risk_limits.max_portfolio_risk)
            stop_loss_distance = kwargs.get("stop_loss_distance", self.risk_limits.stop_loss)
            
            return self.position_sizer.fixed_fractional(
                portfolio_value,
                risk_per_trade,
                stop_loss_distance
            )
            
        elif method == "volatility_based":
            target_volatility = kwargs.get("target_volatility", 0.15)
            
            return self.position_sizer.volatility_based(
                portfolio_value,
                target_volatility,
                volatility
            )
            
        elif method == "equal_weight":
            num_positions = kwargs.get("num_positions", 10)
            
            return self.position_sizer.equal_weight(
                portfolio_value,
                num_positions
            )
            
        else:
            return portfolio_value * self.risk_limits.max_position_size * signal_strength
    
    def check_stop_loss(
        self,
        position: Dict,
        current_price: float
    ) -> bool:
        
        entry_price = position.get("avg_price", position.get("entry_price", 0))
        
        if entry_price <= 0:
            return False
            
        loss_pct = (entry_price - current_price) / entry_price
        
        return loss_pct >= self.risk_limits.stop_loss
    
    def check_take_profit(
        self,
        position: Dict,
        current_price: float
    ) -> bool:
        
        entry_price = position.get("avg_price", position.get("entry_price", 0))
        
        if entry_price <= 0:
            return False
            
        profit_pct = (current_price - entry_price) / entry_price
        
        return profit_pct >= self.risk_limits.take_profit
    
    def update_daily_metrics(
        self,
        portfolio_value: float,
        returns: float,
        positions: Dict
    ):
        
        self.portfolio_history.append(portfolio_value)
        self.daily_returns.append(returns)
        self.positions_history.append(positions.copy())
        
        if len(self.daily_returns) > 252:
            self.daily_returns = self.daily_returns[-252:]
            
        if len(self.portfolio_history) > 252:
            self.portfolio_history = self.portfolio_history[-252:]