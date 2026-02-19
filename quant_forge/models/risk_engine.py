"""
Quant Forge — Financial Mathematics Engine
===========================================
All models implemented from first principles (no black-box wrappers).

References:
  [1] Hull (2018) Options, Futures, and Other Derivatives, 10e
  [2] McNeil, Frey, Embrechts (2015) Quantitative Risk Management
  [3] Hamilton (1994) Time Series Analysis
  [4] Kalman (1960) A New Approach to Linear Filtering and Prediction

Mathematical notation follows standard quant finance conventions.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.linalg import solve

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("quant_forge.models")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometric Brownian Motion
# ─────────────────────────────────────────────────────────────────────────────

class GBM:
    """
    Geometric Brownian Motion: dS_t = μ S_t dt + σ S_t dW_t

    Exact solution (Itô):
        S_t = S_0 · exp[(μ - σ²/2)t + σ√t · Z],  Z ~ N(0,1)

    Parameters
    ----------
    mu    : drift (annualized log-return)
    sigma : volatility (annualized)
    S0    : initial price
    dt    : time step (1/252 for daily)
    T     : horizon in years
    """

    def __init__(self, mu: float, sigma: float, S0: float,
                 dt: float = 1/252, T: float = 1.0):
        self.mu    = mu
        self.sigma = sigma
        self.S0    = S0
        self.dt    = dt
        self.T     = T
        self.N     = int(T / dt)

    def simulate(self, n_paths: int = 10_000, seed: Optional[int] = 42) -> np.ndarray:
        """
        Returns array of shape (n_paths, N+1).

        S_{t+dt} = S_t · exp[(μ - σ²/2)dt + σ√dt · ε],  ε ~ N(0,1)
        """
        rng = np.random.default_rng(seed)
        eps   = rng.standard_normal((n_paths, self.N))
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diff  = self.sigma * np.sqrt(self.dt)
        log_increments = drift + diff * eps            # (n_paths, N)
        log_paths      = np.cumsum(log_increments, axis=1)
        paths = self.S0 * np.exp(np.hstack([
            np.zeros((n_paths, 1)),
            log_paths
        ]))
        return paths                                   # (n_paths, N+1)

    @classmethod
    def calibrate(cls, returns: pd.Series, S0: float) -> "GBM":
        """MLE calibration from log-returns."""
        mu    = returns.mean() * 252
        sigma = returns.std()  * np.sqrt(252)
        return cls(mu=mu, sigma=sigma, S0=S0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Monte Carlo Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    paths:        np.ndarray          # (n_paths, horizon+1)
    final_prices: np.ndarray          # (n_paths,)
    pnl:          np.ndarray          # (n_paths,) dollar PnL
    var_95:       float
    var_99:       float
    es_95:        float
    es_99:        float
    mean_pnl:     float
    std_pnl:      float
    prob_loss:    float


class MonteCarloEngine:
    """
    Portfolio-level Monte Carlo simulation using correlated GBM paths.

    For n assets with covariance Σ, we use Cholesky:
        dS_i = μ_i dt + Σ^{1/2}_{i,·} dW

    VaR_α = -quantile(PnL, 1-α)
    ES_α  = -E[PnL | PnL < -VaR_α]
    """

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray,
                 n_paths: int = 10_000, horizon: int = 252):
        self.returns  = returns
        self.weights  = weights / weights.sum()  # normalize
        self.n_paths  = n_paths
        self.horizon  = horizon
        self.assets   = returns.columns.tolist()
        self.n        = len(self.assets)

    def _estimate_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mu_vec, cov_matrix) annualized."""
        mu  = self.returns.mean().values  * 252
        cov = self.returns.cov().values   * 252
        return mu, cov

    def run(self, S0_vec: Optional[np.ndarray] = None,
            seed: int = 42) -> MonteCarloResult:
        """
        Run correlated multi-asset GBM simulation.
        S0_vec: initial prices (default: 1000 each, portfolio normalized)
        """
        mu, cov = self._estimate_params()
        dt = 1 / 252

        if S0_vec is None:
            S0_vec = np.ones(self.n) * 1000.0

        # Portfolio value at t=0
        V0 = np.dot(self.weights, S0_vec)

        # Cholesky decomposition: Σ = L L^T
        try:
            L = np.linalg.cholesky(cov * dt)
        except np.linalg.LinAlgError:
            cov_reg = cov + np.eye(self.n) * 1e-6
            L = np.linalg.cholesky(cov_reg * dt)

        rng = np.random.default_rng(seed)
        log_returns_paths = np.zeros((self.n_paths, self.horizon, self.n))

        # Batch-generate correlated increments
        Z = rng.standard_normal((self.n_paths, self.horizon, self.n))
        # Correlated: ε = L @ z
        eps = Z @ L.T                                  # (n_paths, horizon, n)
        drift = (mu - 0.5 * np.diag(cov)) * dt        # (n,)
        log_returns_paths = drift + eps

        # Price paths
        log_cum   = np.cumsum(log_returns_paths, axis=1)
        price_rel = np.exp(log_cum)                    # (n_paths, horizon, n)
        final_rel = price_rel[:, -1, :]                # (n_paths, n)

        # Portfolio PnL
        final_prices  = S0_vec * final_rel             # (n_paths, n)
        final_values  = (self.weights * final_prices).sum(axis=1) * V0 / np.dot(self.weights, S0_vec)
        pnl           = final_values - V0

        # Risk metrics
        var_95 = float(-np.percentile(pnl, 5))
        var_99 = float(-np.percentile(pnl, 1))
        es_95  = float(-pnl[pnl <= -var_95].mean())
        es_99  = float(-pnl[pnl <= -var_99].mean())

        return MonteCarloResult(
            paths        = price_rel[:, :, 0],   # first asset for viz
            final_prices = final_prices,
            pnl          = pnl,
            var_95       = var_95,
            var_99       = var_99,
            es_95        = es_95,
            es_99        = es_99,
            mean_pnl     = float(pnl.mean()),
            std_pnl      = float(pnl.std()),
            prob_loss    = float((pnl < 0).mean()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. VaR / ES — Three Methods
# ─────────────────────────────────────────────────────────────────────────────

class RiskMetrics:
    """
    Value-at-Risk (VaR) and Expected Shortfall (ES) via three methods:
      1. Historical simulation (non-parametric)
      2. Parametric (Variance-Covariance, assumes normality)
      3. Monte Carlo (delegated to MonteCarloEngine)

    VaR_α = inf{ x : P(L > x) ≤ 1-α }
    ES_α  = E[L | L > VaR_α]         (Conditional Tail Expectation)
    """

    def __init__(self, returns: pd.Series, confidence: float = 0.95):
        self.returns    = returns
        self.confidence = confidence

    def historical_var(self) -> Tuple[float, float]:
        """
        Historical VaR/ES: order statistics on empirical P&L.
        No distributional assumption.
        """
        losses = -self.returns
        var = float(np.percentile(losses, self.confidence * 100))
        es  = float(losses[losses > var].mean())
        return var, es

    def parametric_var(self) -> Tuple[float, float]:
        """
        Parametric (Delta-Normal) VaR:
          VaR_α = μ_L + z_α · σ_L
        where z_α = Φ^{-1}(α), μ_L = -μ_r, σ_L = σ_r

        ES_α = μ_L + σ_L · φ(z_α)/(1-α)
        """
        mu  = -self.returns.mean()
        sig = self.returns.std()
        z   = stats.norm.ppf(self.confidence)
        var = mu + z * sig
        es  = mu + sig * stats.norm.pdf(z) / (1 - self.confidence)
        return float(var), float(es)

    def cornish_fisher_var(self) -> Tuple[float, float]:
        """
        Cornish-Fisher expansion for non-normal returns:
          z_cf = z + (z²-1)γ/6 + (z³-3z)κ/24 - (2z³-5z)γ²/36
        where γ = skewness, κ = excess kurtosis.
        """
        mu  = self.returns.mean()
        sig = self.returns.std()
        gamma = stats.skew(self.returns)
        kappa = stats.kurtosis(self.returns)    # excess kurtosis
        z     = stats.norm.ppf(1 - self.confidence)
        z_cf  = (z + (z**2 - 1)*gamma/6
                   + (z**3 - 3*z)*kappa/24
                   - (2*z**3 - 5*z)*gamma**2/36)
        var   = -(mu + sig * z_cf)
        # Approximate ES via numerical integration
        xs    = np.linspace(z_cf, z_cf - 5, 1000)
        phi   = stats.norm.pdf(xs)
        es    = -(mu + sig * np.trapz(xs * phi, xs) / (np.trapz(phi, xs) + 1e-12))
        return float(var), float(es)

    def all_methods(self) -> Dict:
        h_var, h_es = self.historical_var()
        p_var, p_es = self.parametric_var()
        c_var, c_es = self.cornish_fisher_var()
        return {
            "historical":   {"var": h_var, "es": h_es},
            "parametric":   {"var": p_var, "es": p_es},
            "cornish_fisher": {"var": c_var, "es": c_es},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Performance Ratios
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMetrics:
    """
    Sharpe, Sortino, Calmar, Omega, Max Drawdown.

    Sharpe  = (E[r] - rf) / σ_r  · √252
    Sortino = (E[r] - rf) / σ_d  · √252,  σ_d = downside std
    Calmar  = CAGR / |MaxDrawdown|
    """

    def __init__(self, returns: pd.Series, rf: float = 0.07 / 252):
        self.r  = returns
        self.rf = rf

    @property
    def excess(self) -> pd.Series:
        return self.r - self.rf

    def sharpe(self) -> float:
        return float(self.excess.mean() / self.excess.std() * np.sqrt(252))

    def sortino(self) -> float:
        downside = self.excess[self.excess < 0]
        sigma_d  = np.sqrt((downside**2).mean())
        return float(self.excess.mean() / (sigma_d + 1e-12) * np.sqrt(252))

    def max_drawdown(self) -> float:
        """MDD = min[(S_t - running_max) / running_max]."""
        cumret  = (1 + self.r).cumprod()
        rolling = cumret.cummax()
        dd      = (cumret - rolling) / rolling
        return float(dd.min())

    def calmar(self) -> float:
        cagr = (1 + self.r).prod() ** (252 / len(self.r)) - 1
        mdd  = abs(self.max_drawdown())
        return float(cagr / (mdd + 1e-12))

    def omega(self, threshold: float = 0.0) -> float:
        """
        Omega Ratio = E[max(r-L,0)] / E[max(L-r,0)]
        L = threshold (minimum acceptable return).
        """
        gains  = np.maximum(self.r - threshold, 0).mean()
        losses = np.maximum(threshold - self.r, 0).mean()
        return float(gains / (losses + 1e-12))

    def cagr(self) -> float:
        return float((1 + self.r).prod() ** (252 / len(self.r)) - 1)

    def annualized_vol(self) -> float:
        return float(self.r.std() * np.sqrt(252))

    def summary(self) -> Dict:
        return {
            "cagr":        round(self.cagr(), 4),
            "ann_vol":     round(self.annualized_vol(), 4),
            "sharpe":      round(self.sharpe(), 4),
            "sortino":     round(self.sortino(), 4),
            "calmar":      round(self.calmar(), 4),
            "omega":       round(self.omega(), 4),
            "max_drawdown":round(self.max_drawdown(), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CAPM
# ─────────────────────────────────────────────────────────────────────────────

class CAPM:
    """
    Capital Asset Pricing Model:
      E[r_i] = r_f + β_i (E[r_m] - r_f)

      β_i = Cov(r_i, r_m) / Var(r_m)
      α_i = r_i - [r_f + β_i (r_m - r_f)]   (Jensen's alpha)

    Uses OLS: r_i - r_f = α + β(r_m - r_f) + ε
    """

    def __init__(self, asset_returns: pd.DataFrame,
                 market_returns: pd.Series, rf: float = 0.07 / 252):
        self.asset_r  = asset_returns
        self.market_r = market_returns
        self.rf       = rf
        self._results: Optional[Dict] = None

    def _align(self) -> Tuple[pd.DataFrame, pd.Series]:
        idx = self.asset_r.index.intersection(self.market_r.index)
        return self.asset_r.loc[idx], self.market_r.loc[idx]

    def fit(self) -> Dict:
        assets, mkt = self._align()
        excess_mkt  = mkt - self.rf
        results     = {}

        for col in assets.columns:
            excess_asset = assets[col] - self.rf
            # OLS: x = excess_mkt, y = excess_asset
            x = excess_mkt.values.reshape(-1, 1)
            y = excess_asset.values

            # β = Cov(y,x) / Var(x)
            beta  = float(np.cov(y, x.ravel())[0, 1] / np.var(x.ravel()))
            alpha = float(y.mean() - beta * x.ravel().mean())
            r2    = float(1 - np.var(y - alpha - beta * x.ravel()) / np.var(y))

            expected_r = self.rf * 252 + beta * (mkt.mean() * 252 - self.rf * 252)
            results[col] = {
                "alpha":      round(alpha * 252, 6),    # annualized
                "beta":       round(beta, 4),
                "r_squared":  round(r2, 4),
                "expected_r": round(expected_r, 4),
                "treynor":    round((assets[col].mean() * 252 - self.rf * 252) / (beta + 1e-12), 4),
            }
        self._results = results
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. GARCH(1,1) Volatility Model
# ─────────────────────────────────────────────────────────────────────────────

class GARCH:
    """
    GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Stationarity: α + β < 1
    Long-run variance: σ̄² = ω / (1 - α - β)

    Estimated via Maximum Likelihood using arch library.
    """

    def __init__(self, returns: pd.Series):
        self.returns = returns * 100           # scale for numerical stability
        self.model_  = None
        self.result_ = None

    def fit(self):
        from arch import arch_model
        self.model_  = arch_model(
            self.returns, vol="Garch", p=1, q=1,
            mean="Constant", dist="normal"
        )
        self.result_ = self.model_.fit(disp="off", show_warning=False)
        return self

    def conditional_vol(self) -> pd.Series:
        """Returns conditional volatility series (annualized)."""
        if self.result_ is None:
            self.fit()
        cond_vol = self.result_.conditional_volatility / 100 * np.sqrt(252)
        cond_vol.index = self.returns.index
        return cond_vol

    def forecast_vol(self, horizon: int = 5) -> np.ndarray:
        """h-step-ahead variance forecast via GARCH recursion."""
        if self.result_ is None:
            self.fit()
        fc = self.result_.forecast(horizon=horizon, reindex=False)
        return np.sqrt(fc.variance.values[-1]) / 100 * np.sqrt(252)

    def params(self) -> Dict:
        if self.result_ is None:
            self.fit()
        p = self.result_.params
        return {
            "omega": round(float(p.get("omega", 0)), 8),
            "alpha": round(float(p.get("alpha[1]", 0)), 6),
            "beta":  round(float(p.get("beta[1]", 0)),  6),
            "persistence": round(float(p.get("alpha[1]", 0) + p.get("beta[1]", 0)), 6),
            "long_run_vol": round(float(np.sqrt(
                p.get("omega", 1e-6) / max(1e-12, 1 - p.get("alpha[1]", 0) - p.get("beta[1]", 0))
            ) / 100 * np.sqrt(252)), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Kalman Filter — Price Trend Smoothing
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    """
    Linear Kalman Filter for price-trend extraction.

    State: x_t = [level, trend]^T  (2D)
    Transition: x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q)
    Observation: y_t = H x_t + v_t,       v_t ~ N(0, R)

    F = [[1, dt], [0, 1]]   (constant-velocity model)
    H = [1, 0]              (observe price only)

    Kalman Gain: K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
    Update:      x_{t|t} = x_{t|t-1} + K_t (y_t - H x_{t|t-1})
    """

    def __init__(self, process_noise: float = 1e-4, obs_noise: float = 0.1):
        self.q = process_noise
        self.r = obs_noise

    def smooth(self, prices: pd.Series) -> pd.DataFrame:
        """
        Returns DataFrame with columns: [smoothed, trend, innovation].
        """
        y = prices.values
        n = len(y)

        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = self.q * np.eye(2)
        R = np.array([[self.r]])

        x = np.array([[y[0]], [0.0]])
        P = np.eye(2) * 1.0

        smoothed   = np.zeros(n)
        trend      = np.zeros(n)
        innovations = np.zeros(n)

        for t in range(n):
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Innovation
            innov = y[t] - (H @ x_pred)[0, 0]

            # Kalman gain
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Update
            x = x_pred + K * innov
            P = (np.eye(2) - K @ H) @ P_pred

            smoothed[t]    = x[0, 0]
            trend[t]       = x[1, 0]
            innovations[t] = innov

        return pd.DataFrame({
            "price":      y,
            "smoothed":   smoothed,
            "trend":      trend,
            "innovation": innovations,
        }, index=prices.index)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PCA Factor Model
# ─────────────────────────────────────────────────────────────────────────────

class PCAFactorModel:
    """
    Principal Component Analysis on return covariance.

    Spectral decomposition: Σ = V Λ V^T
    Factor returns: F = R V_k  (k principal components)
    Explained variance: Λ_i / sum(Λ)
    """

    def __init__(self, n_components: int = 5):
        self.k       = n_components
        self.eigvecs = None
        self.eigvals = None
        self.mean_r  = None

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        from sklearn.decomposition import PCA
        self.mean_r  = returns.mean()
        X            = (returns - self.mean_r).values
        pca          = PCA(n_components=self.k)
        pca.fit(X)
        self.eigvecs           = pca.components_       # (k, n_assets)
        self.eigvals           = pca.explained_variance_
        self.explained_ratio_  = pca.explained_variance_ratio_
        self._pca              = pca
        return self

    def factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        X  = (returns - self.mean_r).values
        F  = X @ self.eigvecs.T
        return pd.DataFrame(F, index=returns.index,
                            columns=[f"PC{i+1}" for i in range(self.k)])

    def reconstruct(self, returns: pd.DataFrame) -> pd.DataFrame:
        F   = self.factor_returns(returns).values
        R_hat = F @ self.eigvecs + self.mean_r.values
        return pd.DataFrame(R_hat, index=returns.index, columns=returns.columns)

    def explained_variance_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "eigenvalue":  self.eigvals,
            "var_ratio":   self.explained_ratio_,
            "cumulative":  np.cumsum(self.explained_ratio_),
        }, index=[f"PC{i+1}" for i in range(self.k)])


# ─────────────────────────────────────────────────────────────────────────────
# 9. Bayesian VaR (via scipy — lightweight PyMC alternative)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianVaR:
    """
    Bayesian estimation of VaR via t-distribution.
    Posterior predictive: model returns as Normal-InverseGamma.

    p(μ, σ² | data) ∝ N(μ | μ_n, σ²/κ_n) · InvGamma(σ² | αn, βn)

    With conjugate NIG priors:
      μ_0 = 0, κ_0 = 1, α_0 = 1, β_0 = 0.01
    """

    def __init__(self, returns: pd.Series):
        self.r  = returns.dropna().values
        self.n  = len(self.r)

    def _posterior_params(self):
        # Hyperpriors
        mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 2.0, 0.01

        # Sample stats
        x_bar = self.r.mean()
        S     = self.r.var() * self.n

        # Posterior updates (Normal-InvGamma conjugate)
        kappa_n = kappa_0 + self.n
        mu_n    = (kappa_0 * mu_0 + self.n * x_bar) / kappa_n
        alpha_n = alpha_0 + self.n / 2
        beta_n  = (beta_0
                   + 0.5 * S
                   + 0.5 * kappa_0 * self.n * (x_bar - mu_0)**2 / kappa_n)
        return mu_n, kappa_n, alpha_n, beta_n

    def posterior_var(self, confidence: float = 0.95,
                      n_samples: int = 20_000) -> Dict:
        """
        Sample from posterior predictive (Student-t):
          predictive = Student-t(2α_n, μ_n, β_n(1+1/κ_n)/α_n)
        """
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_params()

        df_t   = 2 * alpha_n
        scale  = np.sqrt(beta_n * (1 + 1/kappa_n) / alpha_n)
        rng    = np.random.default_rng(0)
        samples = stats.t.rvs(df_t, loc=mu_n, scale=scale,
                              size=n_samples, random_state=0)
        losses  = -samples
        var     = float(np.percentile(losses, confidence * 100))
        es      = float(losses[losses > var].mean())

        return {
            "posterior_var": round(var, 6),
            "posterior_es":  round(es,  6),
            "posterior_mu":  round(mu_n, 6),
            "posterior_sigma": round(scale, 6),
            "df":            round(df_t, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Integrated Risk Report
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_risk_report(
    prices:   pd.DataFrame,
    returns:  pd.DataFrame,
    weights:  np.ndarray,
    rf:       float = 0.07 / 252
) -> Dict:
    """
    Single-call risk engine.
    Returns comprehensive dict with all risk metrics.
    """
    tickers = prices.columns.tolist()

    # Portfolio returns
    port_r = (returns * weights).sum(axis=1)

    # 1. Performance
    perf = PerformanceMetrics(port_r, rf=rf).summary()

    # 2. VaR (per-asset + portfolio)
    var_results = {}
    for t in tickers:
        rm = RiskMetrics(returns[t])
        var_results[t] = rm.all_methods()
    port_rm = RiskMetrics(port_r)
    var_results["PORTFOLIO"] = port_rm.all_methods()

    # 3. Monte Carlo
    mc_engine = MonteCarloEngine(returns, weights, n_paths=10_000)
    mc_result  = mc_engine.run()

    # 4. GARCH per asset
    garch_params = {}
    for t in tickers:
        try:
            g = GARCH(returns[t]).fit()
            garch_params[t] = g.params()
        except Exception as e:
            log.warning(f"GARCH failed for {t}: {e}")

    # 5. Kalman on portfolio price
    port_price = (prices * weights).sum(axis=1)
    kf_result  = KalmanFilter().smooth(port_price)

    # 6. Bayesian VaR
    bayes_var = BayesianVaR(port_r).posterior_var()

    # 7. Correlation matrix
    corr = returns.corr()

    return {
        "tickers":     tickers,
        "weights":     dict(zip(tickers, weights.round(4))),
        "performance": perf,
        "var":         var_results,
        "monte_carlo": {
            "var_95": round(mc_result.var_95, 4),
            "var_99": round(mc_result.var_99, 4),
            "es_95":  round(mc_result.es_95, 4),
            "es_99":  round(mc_result.es_99, 4),
            "mean_pnl":   round(mc_result.mean_pnl, 4),
            "prob_loss":  round(mc_result.prob_loss, 4),
        },
        "garch":       garch_params,
        "kalman":      kf_result,
        "bayesian_var": bayes_var,
        "correlation": corr,
        "mc_object":   mc_result,   # for plotting
    }
