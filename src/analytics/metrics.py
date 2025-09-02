import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class PerformanceReport:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    expectancy: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    trades_count: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    recovery_factor: float
    ulcer_index: float
    skewness: float
    kurtosis: float


class PerformanceMetrics:
    
    def __init__(self):
        self.trading_days = 252
        
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = 0.02,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        
        metrics = {}
        
        metrics.update(self.calculate_return_metrics(returns, equity_curve, initial_capital))
        
        metrics.update(self.calculate_risk_metrics(returns, equity_curve))
        
        metrics.update(self.calculate_risk_adjusted_metrics(returns, risk_free_rate))
        
        if not trades.empty:
            metrics.update(self.calculate_trade_metrics(trades))
            
        if benchmark_returns is not None:
            metrics.update(self.calculate_relative_metrics(returns, benchmark_returns, risk_free_rate))
            
        metrics.update(self.calculate_distribution_metrics(returns))
        
        return metrics
    
    def calculate_return_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        trading_days_in_period = len(returns)
        years = trading_days_in_period / self.trading_days
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
            
        cumulative_returns = (1 + returns).cumprod() - 1
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "cumulative_return": cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0,
            "average_daily_return": returns.mean(),
            "compound_annual_growth_rate": annualized_return
        }
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        
        volatility = returns.std() * np.sqrt(self.trading_days)
        
        drawdown, max_dd, max_dd_duration = self.calculate_drawdown_series(equity_curve)
        
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0
        
        ulcer_index = self.calculate_ulcer_index(drawdown)
        
        return {
            "volatility": volatility,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "downside_deviation": downside_deviation,
            "ulcer_index": ulcer_index,
            "semi_variance": downside_returns.var() if len(downside_returns) > 0 else 0
        }
    
    def calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        
        excess_returns = returns - risk_free_rate / self.trading_days
        
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        
        sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
        
        calmar_ratio = self.calculate_calmar_ratio(returns, risk_free_rate)
        
        omega_ratio = self.calculate_omega_ratio(returns, risk_free_rate / self.trading_days)
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "omega_ratio": omega_ratio,
            "adjusted_sharpe_ratio": sharpe_ratio * np.sqrt(self.trading_days / len(returns)) if len(returns) > 0 else 0
        }
    
    def calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        
        if trades.empty:
            return {}
            
        winning_trades = trades[trades["pnl"] > 0]
        losing_trades = trades[trades["pnl"] < 0]
        
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0
        
        profit_factor = (
            winning_trades["pnl"].sum() / abs(losing_trades["pnl"].sum())
            if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0
            else float("inf") if len(winning_trades) > 0 else 0
        )
        
        expectancy = trades["pnl"].mean()
        
        best_trade = trades["pnl"].max() if len(trades) > 0 else 0
        worst_trade = trades["pnl"].min() if len(trades) > 0 else 0
        
        if "timestamp" in trades.columns:
            trades["duration"] = trades["timestamp"].diff()
            avg_duration = trades["duration"].mean().total_seconds() / 3600 if len(trades) > 1 else 0
        else:
            avg_duration = 0
            
        consecutive_wins = self.calculate_consecutive_wins(trades)
        consecutive_losses = self.calculate_consecutive_losses(trades)
        
        return {
            "trades_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_trade_duration_hours": avg_duration,
            "max_consecutive_wins": consecutive_wins,
            "max_consecutive_losses": consecutive_losses,
            "payoff_ratio": avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0
        }
    
    def calculate_relative_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        
        aligned_returns, aligned_benchmark = self.align_series(returns, benchmark_returns)
        
        if len(aligned_returns) < 2:
            return {}
            
        beta, alpha = self.calculate_beta_alpha(aligned_returns, aligned_benchmark, risk_free_rate)
        
        tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(self.trading_days)
        
        information_ratio = (
            (aligned_returns - aligned_benchmark).mean() / (aligned_returns - aligned_benchmark).std() * np.sqrt(self.trading_days)
            if (aligned_returns - aligned_benchmark).std() > 0 else 0
        )
        
        correlation = aligned_returns.corr(aligned_benchmark)
        
        active_return = aligned_returns.mean() - aligned_benchmark.mean()
        
        return {
            "beta": beta,
            "alpha": alpha,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
            "correlation": correlation,
            "active_return": active_return * self.trading_days,
            "treynor_ratio": (aligned_returns.mean() - risk_free_rate / self.trading_days) / beta if beta != 0 else 0
        }
    
    def calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        
        skewness = returns.skew() if len(returns) > 2 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0
        
        jarque_bera_stat, jarque_bera_pvalue = stats.jarque_bera(returns) if len(returns) > 2 else (0, 1)
        
        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "jarque_bera_stat": jarque_bera_stat,
            "jarque_bera_pvalue": jarque_bera_pvalue,
            "is_normal_distribution": jarque_bera_pvalue > 0.05
        }
    
    def calculate_drawdown_series(
        self,
        equity_curve: pd.Series
    ) -> Tuple[pd.Series, float, int]:
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
                
        return drawdown, max_drawdown, max_duration
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        
        var = self.calculate_var(returns, confidence_level)
        cvar_returns = returns[returns <= var]
        
        return cvar_returns.mean() if len(cvar_returns) > 0 else var
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / self.trading_days
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        target_return: float = 0
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / self.trading_days
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days)
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        annualized_return = returns.mean() * self.trading_days
        equity_curve = (1 + returns).cumprod()
        _, max_drawdown, _ = self.calculate_drawdown_series(equity_curve)
        
        if max_drawdown == 0:
            return 0.0
            
        return (annualized_return - risk_free_rate) / abs(max_drawdown)
    
    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0
    ) -> float:
        
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
            
        return gains / losses
    
    def calculate_ulcer_index(self, drawdown: pd.Series) -> float:
        
        if len(drawdown) < 2:
            return 0.0
            
        squared_drawdowns = drawdown ** 2
        return np.sqrt(squared_drawdowns.mean())
    
    def calculate_beta_alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Tuple[float, float]:
        
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0
            
        excess_returns = returns - risk_free_rate / self.trading_days
        excess_benchmark = benchmark_returns - risk_free_rate / self.trading_days
        
        covariance = excess_returns.cov(excess_benchmark)
        benchmark_variance = excess_benchmark.var()
        
        if benchmark_variance == 0:
            return 0.0, 0.0
            
        beta = covariance / benchmark_variance
        alpha = (excess_returns.mean() - beta * excess_benchmark.mean()) * self.trading_days
        
        return beta, alpha
    
    def calculate_consecutive_wins(self, trades: pd.DataFrame) -> int:
        
        if trades.empty or "pnl" not in trades.columns:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trades["pnl"]:
            if pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def calculate_consecutive_losses(self, trades: pd.DataFrame) -> int:
        
        if trades.empty or "pnl" not in trades.columns:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trades["pnl"]:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def align_series(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        
        common_index = series1.index.intersection(series2.index)
        
        return series1.loc[common_index], series2.loc[common_index]