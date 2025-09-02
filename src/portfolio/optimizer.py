import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.optimize import minimize
import cvxpy as cp
from sklearn.covariance import LedoitWolf, EmpiricalCovariance


class OptimizationObjective(Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_REVERSION = "mean_reversion"
    MINIMUM_CORRELATION = "min_correlation"


@dataclass
class OptimizationConstraints:
    max_weight: float = 0.3
    min_weight: float = 0.0
    max_turnover: float = 1.0
    max_leverage: float = 1.0
    sector_max_weight: Optional[Dict[str, float]] = None
    max_concentration: float = 0.4
    min_assets: int = 5
    max_assets: int = 50
    transaction_cost: float = 0.001
    holding_cost: float = 0.0001


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    objective_value: float
    optimization_success: bool
    risk_contribution: Dict[str, float]
    metadata: Dict[str, Any]


class PortfolioOptimizer:
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        regularization_factor: float = 0.001,
        use_robust_covariance: bool = True
    ):
        self.risk_free_rate = risk_free_rate
        self.regularization_factor = regularization_factor
        self.use_robust_covariance = use_robust_covariance
        self.logger = logging.getLogger(__name__)
        
    def optimize(
        self,
        returns: pd.DataFrame,
        objective: OptimizationObjective,
        constraints: OptimizationConstraints = None,
        current_weights: Optional[Dict[str, float]] = None,
        expected_returns: Optional[pd.Series] = None,
        views: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> OptimizationResult:
        
        constraints = constraints or OptimizationConstraints()
        
        returns_clean = self._prepare_returns(returns)
        
        if returns_clean.empty or len(returns_clean.columns) < 2:
            return self._create_empty_result()
            
        expected_ret = expected_returns or self._estimate_expected_returns(returns_clean)
        cov_matrix = self._estimate_covariance_matrix(returns_clean)
        
        if objective == OptimizationObjective.MAX_SHARPE:
            result = self._max_sharpe_optimization(expected_ret, cov_matrix, constraints)
        elif objective == OptimizationObjective.MIN_VARIANCE:
            result = self._min_variance_optimization(cov_matrix, constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            result = self._risk_parity_optimization(cov_matrix, constraints)
        elif objective == OptimizationObjective.BLACK_LITTERMAN:
            result = self._black_litterman_optimization(
                returns_clean, expected_ret, cov_matrix, views, constraints
            )
        else:
            result = self._max_sharpe_optimization(expected_ret, cov_matrix, constraints)
            
        result.risk_contribution = self._calculate_risk_contribution(
            result.weights, cov_matrix
        )
        
        return result
    
    def _prepare_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        
        returns_clean = returns.copy()
        
        returns_clean = returns_clean.dropna(thresh=int(len(returns_clean) * 0.8), axis=1)
        
        returns_clean = returns_clean.fillna(returns_clean.mean())
        
        returns_clean = returns_clean.select_dtypes(include=[np.number])
        
        for col in returns_clean.columns:
            returns_clean[col] = self._winsorize(returns_clean[col])
            
        return returns_clean
    
    def _winsorize(self, series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        
        lower, upper = series.quantile([limits[0], limits[1]])
        return series.clip(lower=lower, upper=upper)
    
    def _estimate_expected_returns(
        self,
        returns: pd.DataFrame,
        method: str = "historical"
    ) -> pd.Series:
        
        if method == "historical":
            return returns.mean() * 252
        elif method == "ewm":
            return returns.ewm(span=60).mean().iloc[-1] * 252
        elif method == "capm":
            return self._capm_expected_returns(returns)
        else:
            return returns.mean() * 252
    
    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        
        if self.use_robust_covariance:
            if len(returns) > returns.shape[1] * 2:
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns).covariance_
                return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
            else:
                emp_cov = EmpiricalCovariance()
                cov_matrix = emp_cov.fit(returns).covariance_
                return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
        else:
            cov_matrix = returns.cov() * 252
            
        regularization = np.eye(len(cov_matrix)) * self.regularization_factor
        cov_matrix += regularization
        
        return cov_matrix
    
    def _max_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        
        n_assets = len(expected_returns)
        
        weights = cp.Variable(n_assets)
        
        portfolio_return = weights @ expected_returns.values
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Maximize(portfolio_return - self.risk_free_rate)
        
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= constraints.min_weight,
            weights <= constraints.max_weight,
            portfolio_risk <= 1
        ]
        
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if weights.value is not None:
                optimal_weights = dict(zip(expected_returns.index, weights.value))
                
                portfolio_return_val = np.dot(weights.value, expected_returns.values)
                portfolio_risk_val = np.sqrt(weights.value @ cov_matrix.values @ weights.value)
                sharpe_ratio = (portfolio_return_val - self.risk_free_rate) / portfolio_risk_val
                
                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=portfolio_return_val,
                    expected_volatility=portfolio_risk_val,
                    sharpe_ratio=sharpe_ratio,
                    objective_value=problem.value or 0,
                    optimization_success=True,
                    risk_contribution={},
                    metadata={"solver_status": problem.status}
                )
            else:
                return self._create_equal_weight_result(expected_returns, cov_matrix)
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return self._create_equal_weight_result(expected_returns, cov_matrix)
    
    def _min_variance_optimization(
        self,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        
        n_assets = len(cov_matrix)
        
        weights = cp.Variable(n_assets)
        
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Minimize(portfolio_risk)
        
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= constraints.min_weight,
            weights <= constraints.max_weight
        ]
        
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if weights.value is not None:
                optimal_weights = dict(zip(cov_matrix.index, weights.value))
                
                portfolio_risk_val = np.sqrt(weights.value @ cov_matrix.values @ weights.value)
                
                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=0.0,
                    expected_volatility=portfolio_risk_val,
                    sharpe_ratio=0.0,
                    objective_value=problem.value or 0,
                    optimization_success=True,
                    risk_contribution={},
                    metadata={"solver_status": problem.status}
                )
            else:
                return self._create_equal_weight_result(
                    pd.Series(index=cov_matrix.index), cov_matrix
                )
                
        except Exception as e:
            self.logger.error(f"Min variance optimization failed: {e}")
            return self._create_equal_weight_result(
                pd.Series(index=cov_matrix.index), cov_matrix
            )
    
    def _risk_parity_optimization(
        self,
        cov_matrix: pd.DataFrame,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            marginal_risk = (cov_matrix.values @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            target_risk = portfolio_vol / n_assets
            
            return np.sum((risk_contributions - target_risk) ** 2)
        
        def portfolio_constraint(weights):
            return np.sum(weights) - 1.0
        
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
        
        constraints_scipy = [{"type": "eq", "fun": portfolio_constraint}]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_scipy,
                options={"maxiter": 1000, "ftol": 1e-9}
            )
            
            if result.success:
                optimal_weights = dict(zip(cov_matrix.index, result.x))
                portfolio_risk_val = np.sqrt(result.x @ cov_matrix.values @ result.x)
                
                return OptimizationResult(
                    weights=optimal_weights,
                    expected_return=0.0,
                    expected_volatility=portfolio_risk_val,
                    sharpe_ratio=0.0,
                    objective_value=result.fun,
                    optimization_success=True,
                    risk_contribution={},
                    metadata={"scipy_result": result}
                )
            else:
                return self._create_equal_weight_result(
                    pd.Series(index=cov_matrix.index), cov_matrix
                )
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            return self._create_equal_weight_result(
                pd.Series(index=cov_matrix.index), cov_matrix
            )
    
    def _black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        views: Optional[Dict[str, Tuple[float, float]]],
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        
        if views is None:
            views = {}
            
        tau = 0.025
        
        risk_aversion = self._estimate_risk_aversion(returns)
        
        market_caps = np.ones(len(expected_returns))
        w_market = market_caps / market_caps.sum()
        
        pi = risk_aversion * cov_matrix.values @ w_market
        
        if views:
            P = np.zeros((len(views), len(expected_returns)))
            Q = np.zeros(len(views))
            omega_diag = []
            
            for i, (asset, (view_return, confidence)) in enumerate(views.items()):
                if asset in expected_returns.index:
                    asset_idx = list(expected_returns.index).index(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = view_return
                    omega_diag.append(1.0 / confidence)
                    
            omega = np.diag(omega_diag)
            
            tau_cov = tau * cov_matrix.values
            
            try:
                M1 = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
                M2 = np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(omega) @ Q
                
                bl_expected_returns = np.linalg.inv(M1) @ M2
                bl_cov = np.linalg.inv(M1)
                
            except np.linalg.LinAlgError:
                bl_expected_returns = pi
                bl_cov = tau_cov
        else:
            bl_expected_returns = pi
            bl_cov = tau * cov_matrix.values
            
        bl_expected_returns_series = pd.Series(bl_expected_returns, index=expected_returns.index)
        bl_cov_df = pd.DataFrame(bl_cov, index=cov_matrix.index, columns=cov_matrix.columns)
        
        return self._max_sharpe_optimization(bl_expected_returns_series, bl_cov_df, constraints)
    
    def _estimate_risk_aversion(self, returns: pd.DataFrame) -> float:
        
        market_return = returns.mean(axis=1)
        market_vol = market_return.std()
        excess_return = market_return.mean()
        
        if market_vol > 0:
            return excess_return / (market_vol ** 2)
        else:
            return 3.0
    
    def _calculate_risk_contribution(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        
        weights_array = np.array([weights[asset] for asset in cov_matrix.index])
        
        portfolio_vol = np.sqrt(weights_array @ cov_matrix.values @ weights_array)
        
        if portfolio_vol == 0:
            return {asset: 0.0 for asset in cov_matrix.index}
            
        marginal_risk = (cov_matrix.values @ weights_array) / portfolio_vol
        
        risk_contributions = weights_array * marginal_risk
        
        return dict(zip(cov_matrix.index, risk_contributions))
    
    def _create_equal_weight_result(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> OptimizationResult:
        
        n_assets = len(cov_matrix)
        equal_weight = 1.0 / n_assets
        
        weights = {asset: equal_weight for asset in cov_matrix.index}
        
        weights_array = np.array(list(weights.values()))
        
        if len(expected_returns) > 0:
            portfolio_return = np.dot(weights_array, expected_returns.values)
        else:
            portfolio_return = 0.0
            
        portfolio_risk = np.sqrt(weights_array @ cov_matrix.values @ weights_array)
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            objective_value=0.0,
            optimization_success=False,
            risk_contribution={},
            metadata={"fallback": "equal_weight"}
        )
    
    def _create_empty_result(self) -> OptimizationResult:
        
        return OptimizationResult(
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            objective_value=0.0,
            optimization_success=False,
            risk_contribution={},
            metadata={"error": "insufficient_data"}
        )
    
    def _capm_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        
        market_returns = returns.mean(axis=1)
        expected_returns = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            if len(asset_returns.dropna()) < 20:
                expected_returns[asset] = asset_returns.mean() * 252
                continue
                
            covariance = asset_returns.cov(market_returns)
            market_variance = market_returns.var()
            
            if market_variance > 0:
                beta = covariance / market_variance
                expected_return = self.risk_free_rate + beta * (market_returns.mean() * 252 - self.risk_free_rate)
                expected_returns[asset] = expected_return
            else:
                expected_returns[asset] = asset_returns.mean() * 252
                
        return pd.Series(expected_returns)
    
    def calculate_efficient_frontier(
        self,
        returns: pd.DataFrame,
        num_portfolios: int = 100
    ) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        
        expected_returns = self._estimate_expected_returns(returns)
        cov_matrix = self._estimate_covariance_matrix(returns)
        
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        risks = []
        weights_list = []
        
        for target_ret in target_returns:
            try:
                result = self._optimize_for_target_return(
                    expected_returns, cov_matrix, target_ret
                )
                
                risks.append(result.expected_volatility)
                weights_list.append(result.weights)
                
            except:
                risks.append(np.nan)
                weights_list.append({})
                
        returns_list = target_returns.tolist()
        
        return returns_list, risks, weights_list
    
    def _optimize_for_target_return(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float
    ) -> OptimizationResult:
        
        n_assets = len(expected_returns)
        
        weights = cp.Variable(n_assets)
        
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Minimize(portfolio_risk)
        
        constraints_list = [
            cp.sum(weights) == 1,
            weights @ expected_returns.values == target_return,
            weights >= 0
        ]
        
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        optimal_weights = dict(zip(expected_returns.index, weights.value))
        portfolio_risk_val = np.sqrt(weights.value @ cov_matrix.values @ weights.value)
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=target_return,
            expected_volatility=portfolio_risk_val,
            sharpe_ratio=(target_return - self.risk_free_rate) / portfolio_risk_val,
            objective_value=problem.value,
            optimization_success=problem.status == "optimal",
            risk_contribution={},
            metadata={}
        )