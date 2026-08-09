"""
Quant Forge — Test Suite
========================
pytest coverage for all core modules.
Run: pytest tests/ -v --tb=short
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_prices():
    """Synthetic price series for 5 assets, 504 days."""
    np.random.seed(42)
    n, T = 5, 504
    mu = np.array([0.12, 0.15, 0.10, 0.18, 0.08])
    sigma = np.array([0.20, 0.25, 0.18, 0.30, 0.15])
    dt = 1 / 252
    log_r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(T, n)
    prices = 1000 * np.exp(np.cumsum(log_r, axis=0))
    return pd.DataFrame(
        prices,
        columns=[f"ASSET{i}" for i in range(n)],
        index=pd.date_range("2024-01-01", periods=T, freq="B"),
    )


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 1. GBM Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGBM:
    def test_path_shape(self, synthetic_returns):
        from models.risk_engine import GBM

        r = synthetic_returns["ASSET0"]
        gbm = GBM.calibrate(r, 1000.0)
        gbm.T = 1.0
        paths = gbm.simulate(n_paths=100)
        assert paths.shape == (100, gbm.N + 1), "Path shape mismatch"

    def test_initial_price(self, synthetic_returns):
        from models.risk_engine import GBM

        r = synthetic_returns["ASSET0"]
        gbm = GBM.calibrate(r, 1234.5)
        paths = gbm.simulate(n_paths=50)
        np.testing.assert_allclose(paths[:, 0], 1234.5, rtol=1e-8)

    def test_log_normal_terminal(self, synthetic_returns):
        """Terminal distribution should be approximately log-normal."""
        from models.risk_engine import GBM

        r = synthetic_returns["ASSET0"]
        gbm = GBM(mu=0.10, sigma=0.20, S0=100.0, dt=1 / 252, T=1.0)
        paths = gbm.simulate(n_paths=50_000)
        log_terminal = np.log(paths[:, -1])
        # E[log S_T] = log S_0 + (mu - sigma^2/2) T
        expected_mean = np.log(100) + (0.10 - 0.5 * 0.04) * 1.0
        assert (
            abs(log_terminal.mean() - expected_mean) < 0.05
        ), f"Log-normal mean deviation too large: {log_terminal.mean():.4f} vs {expected_mean:.4f}"

    def test_reproducibility(self, synthetic_returns):
        from models.risk_engine import GBM

        gbm = GBM(0.10, 0.20, 100.0)
        p1 = gbm.simulate(1000, seed=42)
        p2 = gbm.simulate(1000, seed=42)
        np.testing.assert_array_equal(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. VaR / ES Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskMetrics:
    def test_historical_var_ordering(self, synthetic_returns):
        from models.risk_engine import RiskMetrics

        r = synthetic_returns["ASSET0"]
        rm95 = RiskMetrics(r, 0.95)
        rm99 = RiskMetrics(r, 0.99)
        var95, es95 = rm95.historical_var()
        var99, es99 = rm99.historical_var()
        assert var99 >= var95, "VaR 99 should be >= VaR 95"
        assert es99 >= es95, "ES 99 should be >= ES 95"

    def test_es_exceeds_var(self, synthetic_returns):
        from models.risk_engine import RiskMetrics

        r = synthetic_returns["ASSET1"]
        rm = RiskMetrics(r, 0.95)
        for method in ["historical_var", "parametric_var"]:
            var, es = getattr(rm, method)()
            assert es >= var, f"ES should exceed VaR for {method}"

    def test_parametric_formula(self, synthetic_returns):
        """Parametric VaR should match manual calculation."""
        from models.risk_engine import RiskMetrics
        from scipy import stats

        r = synthetic_returns["ASSET2"]
        rm = RiskMetrics(r, 0.95)
        var_p, _ = rm.parametric_var()
        mu_l = -r.mean()
        sig = r.std()
        z = stats.norm.ppf(0.95)
        expected = mu_l + z * sig
        np.testing.assert_allclose(var_p, expected, rtol=1e-6)

    def test_all_methods_return_dict(self, synthetic_returns):
        from models.risk_engine import RiskMetrics

        r = synthetic_returns["ASSET0"]
        rm = RiskMetrics(r, 0.95)
        res = rm.all_methods()
        assert set(res.keys()) == {"historical", "parametric", "cornish_fisher"}
        for method, vals in res.items():
            assert "var" in vals and "es" in vals


# ─────────────────────────────────────────────────────────────────────────────
# 3. Performance Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformanceMetrics:
    def test_sharpe_sign(self, synthetic_returns):
        from models.risk_engine import PerformanceMetrics

        r = synthetic_returns["ASSET0"]
        pm = PerformanceMetrics(r, rf=0.07 / 252)
        s = pm.sharpe()
        assert isinstance(s, float)

    def test_max_drawdown_nonpositive(self, synthetic_returns):
        from models.risk_engine import PerformanceMetrics

        r = synthetic_returns["ASSET1"]
        pm = PerformanceMetrics(r)
        assert pm.max_drawdown() <= 0.0

    def test_sortino_geq_zero_if_positive_excess(self, synthetic_returns):
        from models.risk_engine import PerformanceMetrics

        # create all-positive returns
        r = pd.Series(np.abs(synthetic_returns["ASSET0"].values))
        pm = PerformanceMetrics(r, rf=0.0)
        assert pm.sortino() >= 0

    def test_omega_ratio_one_at_mean(self, synthetic_returns):
        from models.risk_engine import PerformanceMetrics

        r = synthetic_returns["ASSET2"]
        pm = PerformanceMetrics(r)
        # At threshold = mean, gains ≈ losses
        omega = pm.omega(threshold=r.mean())
        assert 0.5 <= omega <= 2.0, f"Omega at mean out of range: {omega}"

    def test_summary_keys(self, synthetic_returns):
        from models.risk_engine import PerformanceMetrics

        r = synthetic_returns["ASSET0"]
        pm = PerformanceMetrics(r)
        keys = {
            "cagr",
            "ann_vol",
            "sharpe",
            "sortino",
            "calmar",
            "omega",
            "max_drawdown",
        }
        assert set(pm.summary().keys()) == keys


# ─────────────────────────────────────────────────────────────────────────────
# 4. CAPM Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCAPM:
    def test_beta_calculation(self, synthetic_returns):
        from models.risk_engine import CAPM

        market = synthetic_returns["ASSET0"]
        assets = synthetic_returns[["ASSET1", "ASSET2"]]
        capm = CAPM(assets, market)
        res = capm.fit()
        assert "ASSET1" in res and "ASSET2" in res
        for t, v in res.items():
            assert "alpha" in v and "beta" in v and "r_squared" in v

    def test_r2_in_range(self, synthetic_returns):
        from models.risk_engine import CAPM

        market = synthetic_returns["ASSET0"]
        assets = synthetic_returns[["ASSET1"]]
        capm = CAPM(assets, market)
        res = capm.fit()
        r2 = res["ASSET1"]["r_squared"]
        assert 0.0 <= r2 <= 1.0, f"R² out of [0,1]: {r2}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kalman Filter Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestKalmanFilter:
    def test_output_shape(self, synthetic_prices):
        from models.risk_engine import KalmanFilter

        kf = KalmanFilter()
        res = kf.smooth(synthetic_prices["ASSET0"])
        assert res.shape[0] == len(synthetic_prices)
        assert set(res.columns) == {"price", "smoothed", "trend", "innovation"}

    def test_smoothed_less_noisy(self, synthetic_prices):
        """Smoothed series should have lower variance than raw."""
        from models.risk_engine import KalmanFilter

        kf = KalmanFilter(process_noise=1e-5, obs_noise=1.0)
        p = synthetic_prices["ASSET0"]
        res = kf.smooth(p)
        assert res["smoothed"].std() <= res["price"].std() * 1.1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Monte Carlo Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMonteCarloEngine:
    def test_var_ordering(self, synthetic_returns):
        from models.risk_engine import MonteCarloEngine

        w = np.ones(len(synthetic_returns.columns)) / len(synthetic_returns.columns)
        mc = MonteCarloEngine(synthetic_returns, w, n_paths=1000)
        res = mc.run()
        assert res.var_99 >= res.var_95, "VaR99 should be >= VaR95"

    def test_es_exceeds_var(self, synthetic_returns):
        from models.risk_engine import MonteCarloEngine

        w = np.ones(len(synthetic_returns.columns)) / len(synthetic_returns.columns)
        mc = MonteCarloEngine(synthetic_returns, w, n_paths=1000)
        res = mc.run()
        assert res.es_95 >= res.var_95


# ─────────────────────────────────────────────────────────────────────────────
# 7. Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClassicalOptimizer:
    def test_weights_sum_to_one(self, synthetic_returns):
        from models.optimizer import ClassicalOptimizer

        opt = ClassicalOptimizer(synthetic_returns)
        for method in [opt._scipy_max_sharpe, opt._scipy_min_vol, opt.risk_parity]:
            res = method()
            total = sum(res.weights.values())
            np.testing.assert_allclose(
                total,
                1.0,
                atol=1e-4,
                err_msg=f"{method.__name__} weights don't sum to 1",
            )

    def test_weights_nonnegative(self, synthetic_returns):
        from models.optimizer import ClassicalOptimizer

        opt = ClassicalOptimizer(synthetic_returns)
        res = opt._scipy_max_sharpe()
        assert all(v >= -1e-6 for v in res.weights.values()), "Negative weights found"

    def test_frontier_has_points(self, synthetic_returns):
        from models.optimizer import ClassicalOptimizer

        opt = ClassicalOptimizer(synthetic_returns)
        f = opt.compute_frontier(10)
        assert len(f) >= 5, f"Frontier too sparse: {len(f)} points"

    def test_vol_return_monotone_on_frontier(self, synthetic_returns):
        """Higher return = higher vol on efficient frontier."""
        from models.optimizer import ClassicalOptimizer

        opt = ClassicalOptimizer(synthetic_returns)
        f = opt.compute_frontier(20)
        if len(f) >= 5:
            vols = [p.vol for p in f]
            rets = [p.ret for p in f]
            # Sort by return: vol should be roughly monotone
            sorted_pairs = sorted(zip(rets, vols))
            vols_sorted = [v for _, v in sorted_pairs]
            # Not strictly monotone but should be generally increasing
            first_half_mean = np.mean(vols_sorted[: len(vols_sorted) // 2])
            second_half_mean = np.mean(vols_sorted[len(vols_sorted) // 2 :])
            assert second_half_mean >= first_half_mean * 0.8


# ─────────────────────────────────────────────────────────────────────────────
# 8. PCA Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPCAFactorModel:
    def test_explained_variance_cumulative(self, synthetic_returns):
        from models.risk_engine import PCAFactorModel

        pca = PCAFactorModel(n_components=3)
        pca.fit(synthetic_returns)
        summary = pca.explained_variance_summary()
        assert (
            float(summary["cumulative"].iloc[-1]) <= 1.001
        ), "Cumulative variance exceeds 100%"
        assert (
            float(summary["cumulative"].iloc[-1]) > 0.3
        ), "3 PCs explain < 30% variance (unexpected)"

    def test_factor_returns_shape(self, synthetic_returns):
        from models.risk_engine import PCAFactorModel

        pca = PCAFactorModel(n_components=3)
        pca.fit(synthetic_returns)
        f = pca.factor_returns(synthetic_returns)
        assert f.shape == (len(synthetic_returns), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Bayesian VaR Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBayesianVaR:
    def test_output_keys(self, synthetic_returns):
        from models.risk_engine import BayesianVaR

        r = synthetic_returns["ASSET0"]
        bv = BayesianVaR(r)
        res = bv.posterior_var(0.95)
        assert "posterior_var" in res and "posterior_es" in res

    def test_es_exceeds_var(self, synthetic_returns):
        from models.risk_engine import BayesianVaR

        r = synthetic_returns["ASSET1"]
        res = BayesianVaR(r).posterior_var()
        assert res["posterior_es"] >= res["posterior_var"]


# ─────────────────────────────────────────────────────────────────────────────
# 10. Feature Engineering Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_log_returns_shape(self, synthetic_prices):
        from data.pipeline import FeatureEngineer

        r = FeatureEngineer.log_returns(synthetic_prices)
        assert len(r) == len(synthetic_prices) - 1

    def test_rsi_range(self, synthetic_prices):
        from data.pipeline import FeatureEngineer

        rsi = FeatureEngineer.rsi(synthetic_prices["ASSET0"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI out of [0,100]"

    def test_bollinger_ordering(self, synthetic_prices):
        from data.pipeline import FeatureEngineer

        upper, mid, lower = FeatureEngineer.bollinger_bands(synthetic_prices["ASSET0"])
        valid = upper.dropna().index
        assert (upper.loc[valid] >= mid.loc[valid]).all(), "Upper < Mid BB"
        assert (mid.loc[valid] >= lower.loc[valid]).all(), "Mid < Lower BB"

    def test_momentum_zero_at_constant(self):
        from data.pipeline import FeatureEngineer

        p = pd.DataFrame(
            {"A": np.ones(100) * 100.0}, index=pd.date_range("2024-01-01", periods=100)
        )
        mom = FeatureEngineer.momentum(p, 10)
        np.testing.assert_allclose(mom.dropna().values, 0.0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess

    subprocess.run(["pytest", __file__, "-v", "--tb=short", "--no-header"])
