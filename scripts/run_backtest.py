#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime
import yaml
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.momentum import MomentumStrategy, CrossSectionalMomentum
from src.strategies.mean_reversion import MeanReversionStrategy
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.data.market_data import DataPipeline, YahooDataProvider
from src.risk.manager import RiskManager
from src.analytics.metrics import PerformanceMetrics


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_strategy(strategy_type: str, config: dict, universe: list):
    if strategy_type == "momentum":
        params = config.get("strategies", {}).get("momentum", {})
        return MomentumStrategy(
            universe=universe,
            lookback_period=params.get("lookback_period", 20),
            num_stocks=params.get("num_stocks", 10),
            rebalance_frequency=params.get("rebalance_frequency", "daily")
        )
    
    elif strategy_type == "cross_sectional_momentum":
        params = config.get("strategies", {}).get("momentum", {})
        return CrossSectionalMomentum(
            universe=universe,
            lookback_period=params.get("lookback_period", 20),
            num_stocks=params.get("num_stocks", 10)
        )
    
    elif strategy_type == "mean_reversion":
        params = config.get("strategies", {}).get("mean_reversion", {})
        return MeanReversionStrategy(
            universe=universe,
            lookback_period=params.get("lookback_period", 20),
            entry_threshold=params.get("entry_threshold", 2.0),
            exit_threshold=params.get("exit_threshold", 0.5)
        )
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--strategy", required=True, 
                       choices=["momentum", "cross_sectional_momentum", "mean_reversion"],
                       help="Strategy to backtest")
    parser.add_argument("--universe", default="sp500",
                       help="Universe to trade (sp500, etfs, crypto)")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100000,
                       help="Initial capital")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--output", default="results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Get universe
    universe = config.get("universes", {}).get(args.universe, [])
    if not universe:
        logger.error(f"Universe '{args.universe}' not found in config")
        sys.exit(1)
    
    logger.info(f"Running {args.strategy} backtest on {args.universe} universe")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Universe size: {len(universe)} assets")
    
    # Create strategy
    strategy = create_strategy(args.strategy, config, universe)
    
    # Create backtest configuration
    backtest_params = config.get("backtesting", {})
    backtest_config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission=backtest_params.get("commission", 0.001),
        slippage=backtest_params.get("slippage", 0.0005),
        benchmark=backtest_params.get("benchmark", "SPY"),
        risk_free_rate=backtest_params.get("risk_free_rate", 0.02)
    )
    
    # Initialize components
    data_pipeline = DataPipeline()
    risk_manager = RiskManager()
    engine = BacktestEngine(strategy, backtest_config, risk_manager)
    
    # Fetch data
    logger.info("Fetching market data...")
    try:
        # For demo purposes, create synthetic data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame(index=dates)
        for symbol in universe[:10]:  # Limit for demo
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)
            data[symbol] = prices
            
        logger.info(f"Data loaded: {data.shape}")
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)
    
    # Run backtest
    logger.info("Running backtest...")
    try:
        results = engine.run(data, start_date, end_date)
        
        # Print results
        logger.info("Backtest completed successfully!")
        logger.info(f"Total Return: {results.metrics.get('total_return', 0):.2%}")
        logger.info(f"Annualized Return: {results.metrics.get('annualized_return', 0):.2%}")
        logger.info(f"Volatility: {results.metrics.get('volatility', 0):.2%}")
        logger.info(f"Sharpe Ratio: {results.metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Max Drawdown: {results.metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Total Trades: {results.metrics.get('trades_count', 0)}")
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        
        results.equity_curve.to_csv(f"{args.output}/equity_curve.csv")
        results.returns.to_csv(f"{args.output}/returns.csv")
        
        if not results.trades.empty:
            results.trades.to_csv(f"{args.output}/trades.csv")
        
        # Save metrics
        import json
        with open(f"{args.output}/metrics.json", 'w') as f:
            json.dump(results.metrics, f, indent=2, default=str)
        
        logger.info(f"Results saved to {args.output}/")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()