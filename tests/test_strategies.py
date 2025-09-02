import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.momentum import MomentumStrategy, CrossSectionalMomentum
from src.strategies.mean_reversion import MeanReversionStrategy, StatisticalArbitrage
from src.strategies.base import Signal, OrderSide, PositionSide


class TestMomentumStrategy:
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)
        
        data = pd.DataFrame(index=dates)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in symbols:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            data[symbol] = prices
            
        return data
    
    @pytest.fixture
    def momentum_strategy(self):
        return MomentumStrategy(
            universe=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            lookback_period=20,
            num_stocks=3
        )
    
    def test_strategy_initialization(self, momentum_strategy):
        assert momentum_strategy.name == "Momentum"
        assert momentum_strategy.lookback_period == 20
        assert momentum_strategy.num_stocks == 3
        assert len(momentum_strategy.universe) == 5
        
    def test_strategy_initialize(self, momentum_strategy):
        initial_capital = 100000
        start_date = datetime(2023, 1, 1)
        
        momentum_strategy.initialize(initial_capital, start_date)
        
        assert momentum_strategy.initial_capital == initial_capital
        assert momentum_strategy.cash == initial_capital
        assert momentum_strategy.start_date == start_date
        assert momentum_strategy.is_initialized == True
        
    @pytest.mark.unit
    def test_calculate_momentum_return(self, momentum_strategy, sample_data):
        momentum_strategy.momentum_metric = "return"
        momentum_scores = momentum_strategy.calculate_momentum(sample_data)
        
        assert len(momentum_scores) == len(sample_data.columns)
        assert not momentum_scores.isnull().any()
        
    @pytest.mark.unit
    def test_generate_signals(self, momentum_strategy, sample_data):
        momentum_strategy.initialize(100000, datetime(2023, 1, 1))
        momentum_strategy.rebalance_counter = 21
        
        signals = momentum_strategy.generate_signals(sample_data)
        
        assert isinstance(signals, list)
        assert len(signals) <= momentum_strategy.num_stocks
        
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.symbol in momentum_strategy.universe
            assert signal.side in [OrderSide.BUY, OrderSide.SELL]
            
    @pytest.mark.unit
    def test_position_sizing_equal_weight(self, momentum_strategy):
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7
        )
        
        position_size = momentum_strategy.calculate_position_size(
            signal, 150.0, 100000
        )
        
        expected_size = 100000 / momentum_strategy.num_stocks / 150.0
        assert abs(position_size - expected_size) < 0.01
        
    @pytest.mark.unit
    def test_should_rebalance_daily(self, momentum_strategy):
        momentum_strategy.rebalance_frequency = "daily"
        assert momentum_strategy.should_rebalance() == True
        
    @pytest.mark.unit
    def test_should_rebalance_weekly(self, momentum_strategy):
        momentum_strategy.rebalance_frequency = "weekly"
        momentum_strategy.rebalance_counter = 5
        assert momentum_strategy.should_rebalance() == True
        
        momentum_strategy.rebalance_counter = 3
        assert momentum_strategy.should_rebalance() == False


class TestCrossSectionalMomentum:
    
    @pytest.fixture
    def cs_momentum_strategy(self):
        return CrossSectionalMomentum(
            universe=["AAPL", "MSFT", "GOOGL"],
            lookback_period=30
        )
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", end="2023-02-28", freq="D")
        np.random.seed(123)
        
        data = pd.DataFrame(index=dates)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for i, symbol in enumerate(symbols):
            trend = 0.002 + i * 0.001
            prices = 100 * np.cumprod(1 + np.random.normal(trend, 0.02, len(dates)))
            data[symbol] = prices
            
        return data
    
    @pytest.mark.unit
    def test_cross_sectional_momentum_calculation(self, cs_momentum_strategy, sample_data):
        momentum_scores = cs_momentum_strategy.calculate_momentum(sample_data)
        
        assert len(momentum_scores) == len(sample_data.columns)
        assert not momentum_scores.isnull().any()
        
        expected_symbols = sample_data.columns.tolist()
        assert all(symbol in expected_symbols for symbol in momentum_scores.index)


class TestMeanReversionStrategy:
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)
        
        data = pd.DataFrame(index=dates)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in symbols:
            base_price = 100
            trend = np.linspace(0, 0.2, len(dates))
            noise = np.random.normal(0, 0.02, len(dates))
            mean_reverting_component = 0.1 * np.sin(np.arange(len(dates)) * 0.1)
            
            prices = base_price * (1 + trend + noise + mean_reverting_component)
            data[symbol] = prices
            
        return data
    
    @pytest.fixture
    def mean_reversion_strategy(self):
        return MeanReversionStrategy(
            universe=["AAPL", "MSFT", "GOOGL"],
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
    
    @pytest.mark.unit
    def test_bollinger_signal_generation(self, mean_reversion_strategy, sample_data):
        mean_reversion_strategy.initialize(100000, datetime(2023, 1, 1))
        
        symbol = "AAPL"
        price_data = sample_data[symbol]
        
        signal = mean_reversion_strategy.generate_bollinger_signal(symbol, price_data)
        
        if signal is not None:
            assert isinstance(signal, Signal)
            assert signal.symbol == symbol
            assert signal.side in [OrderSide.BUY, OrderSide.SELL]
            assert 0 <= signal.strength <= 1
            assert 0 <= signal.confidence <= 1
            
    @pytest.mark.unit
    def test_rsi_calculation(self, mean_reversion_strategy, sample_data):
        price_data = sample_data["AAPL"]
        rsi = mean_reversion_strategy.calculate_rsi(price_data)
        
        assert len(rsi) == len(price_data)
        assert (rsi >= 0).all() and (rsi <= 100).all()
        assert not rsi.isnull().any()
        
    @pytest.mark.unit
    def test_position_sizing_risk_based(self, mean_reversion_strategy):
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            side=OrderSide.BUY,
            strength=0.8,
            confidence=0.7,
            metadata={"z_score": -2.5}
        )
        
        position_size = mean_reversion_strategy.calculate_position_size(
            signal, 150.0, 100000
        )
        
        assert position_size > 0
        max_position = 100000 * 0.1 / 150.0
        assert position_size <= max_position


class TestStatisticalArbitrage:
    
    @pytest.fixture
    def pairs_data(self):
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        np.random.seed(42)
        
        base_trend = np.cumsum(np.random.normal(0.001, 0.01, len(dates)))
        
        data = pd.DataFrame(index=dates)
        data["STOCK_A"] = 100 + base_trend + np.random.normal(0, 0.5, len(dates))
        data["STOCK_B"] = 100 + 0.8 * base_trend + np.random.normal(0, 0.5, len(dates))
        
        return data
    
    @pytest.fixture
    def stat_arb_strategy(self):
        return StatisticalArbitrage(
            universe=["STOCK_A", "STOCK_B"],
            lookback_period=30,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
    
    @pytest.mark.unit
    def test_cointegration_test(self, stat_arb_strategy, pairs_data):
        series1 = pairs_data["STOCK_A"]
        series2 = pairs_data["STOCK_B"]
        
        result = stat_arb_strategy.test_cointegration(series1, series2)
        
        assert isinstance(result, bool)
        
    @pytest.mark.unit
    def test_hedge_ratio_calculation(self, stat_arb_strategy, pairs_data):
        series1 = pairs_data["STOCK_A"]
        series2 = pairs_data["STOCK_B"]
        
        hedge_ratio = stat_arb_strategy.calculate_hedge_ratio(series1, series2)
        
        assert isinstance(hedge_ratio, float)
        assert hedge_ratio > 0
        
    @pytest.mark.unit
    def test_pair_signal_generation(self, stat_arb_strategy, pairs_data):
        stat_arb_strategy.initialize(100000, datetime(2023, 1, 1))
        
        pair = ("STOCK_A", "STOCK_B")
        signals = stat_arb_strategy.generate_pair_signal(pair, pairs_data)
        
        if signals is not None:
            assert isinstance(signals, list)
            assert len(signals) == 2
            
            for signal in signals:
                assert isinstance(signal, Signal)
                assert signal.symbol in pair


@pytest.mark.integration
class TestStrategyIntegration:
    
    def test_momentum_strategy_workflow(self):
        dates = pd.date_range(start="2023-01-01", end="2023-02-28", freq="D")
        np.random.seed(42)
        
        data = pd.DataFrame(index=dates)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        for symbol in symbols:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            data[symbol] = prices
            
        strategy = MomentumStrategy(
            universe=symbols,
            lookback_period=20,
            num_stocks=3
        )
        
        strategy.initialize(100000, dates[0])
        
        orders = strategy.on_data(dates[-1], data)
        
        assert isinstance(orders, list)
        
        for order in orders:
            assert hasattr(order, 'symbol')
            assert hasattr(order, 'side')
            assert hasattr(order, 'quantity')
            
    def test_mean_reversion_strategy_workflow(self):
        dates = pd.date_range(start="2023-01-01", end="2023-02-28", freq="D")
        np.random.seed(123)
        
        data = pd.DataFrame(index=dates)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in symbols:
            base_price = 100
            mean_reverting = 20 * np.sin(np.arange(len(dates)) * 0.2)
            noise = np.random.normal(0, 2, len(dates))
            data[symbol] = base_price + mean_reverting + noise
            
        strategy = MeanReversionStrategy(
            universe=symbols,
            lookback_period=20,
            entry_threshold=1.5,
            exit_threshold=0.5
        )
        
        strategy.initialize(100000, dates[0])
        
        orders = strategy.on_data(dates[-1], data)
        
        assert isinstance(orders, list)
        
        performance_metrics = strategy.get_performance_metrics()
        assert isinstance(performance_metrics, dict)
        assert "total_return" in performance_metrics
        assert "portfolio_value" in performance_metrics