"""
Quant Forge — Portfolio Optimization Engine
===========================================
Classical: Mean-Variance (Markowitz), Max Sharpe, Min Vol via PyPortfolioOpt.
Quantum:   QAOA on QUBO formulation (Qiskit Aer simulator).

Mathematical Framework
----------------------
Markowitz (1952):
  min  w^T Σ w         subject to  w^T μ = μ_target
  s.t. 1^T w = 1,  w ≥ 0

QUBO for portfolio selection:
  min  λ_risk · w^T Σ w  -  λ_ret · μ^T w  +  λ_card · ||w||_0
  Encoded as: H_Q = Σ_{ij} Q_{ij} x_i x_j,  x_i ∈ {0,1}

References:
  [1] Markowitz (1952) J. Finance
  [2] Farhi et al. (2014) QAOA arXiv:1411.4028
  [3] Mugel et al. (2022) Quantum portfolio optimization
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger("quant_forge.optimizer")
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrontierPoint:
    weights:   np.ndarray
    ret:       float
    vol:       float
    sharpe:    float
    labels:    List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    method:      str
    weights:     Dict[str, float]
    expected_ret: float
    expected_vol: float
    sharpe:      float
    frontier:    Optional[List[FrontierPoint]] = None
    metadata:    Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical Optimizer (Markowitz + PyPortfolioOpt)
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalOptimizer:
    """
    Full Efficient Frontier suite:
      - Max Sharpe Ratio (tangency portfolio)
      - Minimum Volatility
      - Custom target return
      - Black-Litterman (with views)
      - Risk Parity
    """

    def __init__(self, returns: pd.DataFrame, rf: float = 0.07 / 252):
        self.returns = returns
        self.rf_ann  = rf * 252
        self.mu_ann  = returns.mean() * 252
        self.cov_ann = returns.cov()  * 252
        self.tickers = returns.columns.tolist()
        self.n       = len(self.tickers)

    # ── PyPortfolioOpt wrappers ───────────────────────────────────────────────

    def max_sharpe(self, l2_reg: float = 0.1) -> OptimizationResult:
        """Maximize Sharpe via convex optimization with L2 regularization."""
        try:
            from pypfopt import EfficientFrontier, expected_returns, risk_models
            mu  = expected_returns.mean_historical_return(
                      pd.DataFrame({t: (1+self.returns[t]).cumprod() for t in self.tickers}))
            cov = risk_models.CovarianceShrinkage(
                      pd.DataFrame({t: (1+self.returns[t]).cumprod() for t in self.tickers})).ledoit_wolf()
            ef  = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef.add_objective(__import__('pypfopt').objective_functions.L2_reg, gamma=l2_reg)
            ef.max_sharpe(risk_free_rate=self.rf_ann)
            w   = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=self.rf_ann, verbose=False)
            return OptimizationResult(
                method       = "max_sharpe_pyportopt",
                weights      = dict(w),
                expected_ret = round(perf[0], 4),
                expected_vol = round(perf[1], 4),
                sharpe       = round(perf[2], 4),
            )
        except Exception as e:
            log.warning(f"PyPortfolioOpt max_sharpe failed: {e}, using scipy")
            return self._scipy_max_sharpe()

    def min_volatility(self) -> OptimizationResult:
        """Global Minimum Variance Portfolio."""
        try:
            from pypfopt import EfficientFrontier, expected_returns, risk_models
            mu  = self.mu_ann
            cov = self.cov_ann
            ef  = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef.min_volatility()
            w    = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=self.rf_ann, verbose=False)
            return OptimizationResult(
                method       = "min_vol_pyportopt",
                weights      = dict(w),
                expected_ret = round(perf[0], 4),
                expected_vol = round(perf[1], 4),
                sharpe       = round(perf[2], 4),
            )
        except Exception as e:
            log.warning(f"PyPortfolioOpt min_vol failed: {e}, using scipy")
            return self._scipy_min_vol()

    # ── Scipy implementations (fallback + custom) ────────────────────────────

    def _scipy_max_sharpe(self) -> OptimizationResult:
        mu  = self.mu_ann.values
        cov = self.cov_ann.values
        rf  = self.rf_ann
        n   = self.n

        def neg_sharpe(w):
            r = w @ mu
            v = np.sqrt(w @ cov @ w)
            return -(r - rf) / (v + 1e-12)

        bounds      = [(0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        w0 = np.ones(n) / n
        res = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
        w = np.maximum(res.x, 0); w /= w.sum()
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        return OptimizationResult(
            method="max_sharpe_scipy",
            weights=dict(zip(self.tickers, w.round(6))),
            expected_ret=round(r, 4),
            expected_vol=round(v, 4),
            sharpe=round((r - rf) / (v + 1e-12), 4),
        )

    def _scipy_min_vol(self) -> OptimizationResult:
        mu  = self.mu_ann.values
        cov = self.cov_ann.values
        n   = self.n

        def portfolio_vol(w):
            return np.sqrt(w @ cov @ w)

        bounds      = [(0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        w0 = np.ones(n) / n
        res = minimize(portfolio_vol, w0, bounds=bounds, constraints=constraints,
                       method="SLSQP")
        w = np.maximum(res.x, 0); w /= w.sum()
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        return OptimizationResult(
            method="min_vol_scipy",
            weights=dict(zip(self.tickers, w.round(6))),
            expected_ret=round(r, 4),
            expected_vol=round(v, 4),
            sharpe=round((r - self.rf_ann) / (v + 1e-12), 4),
        )

    def risk_parity(self) -> OptimizationResult:
        """
        Risk Parity: equalizes marginal risk contributions.
        RC_i = w_i (∂σ_p/∂w_i)^T = σ_p / n  for all i

        Solved via non-linear optimization:
          min Σ (RC_i - σ/n)²
        """
        cov = self.cov_ann.values
        mu  = self.mu_ann.values
        n   = self.n

        def risk_budget_obj(w):
            sig  = np.sqrt(w @ cov @ w)
            MRC  = cov @ w / (sig + 1e-12)   # marginal risk contrib
            RC   = w * MRC                    # risk contrib
            target = sig / n
            return np.sum((RC - target)**2)

        bounds      = [(1e-4, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        w0 = np.ones(n) / n
        res = minimize(risk_budget_obj, w0, bounds=bounds, constraints=constraints,
                       method="SLSQP", options={"maxiter": 5000})
        w = np.maximum(res.x, 0); w /= w.sum()
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        return OptimizationResult(
            method="risk_parity",
            weights=dict(zip(self.tickers, w.round(6))),
            expected_ret=round(r, 4),
            expected_vol=round(v, 4),
            sharpe=round((r - self.rf_ann) / (v + 1e-12), 4),
        )

    def compute_frontier(self, n_points: int = 50) -> List[FrontierPoint]:
        """
        Trace the Efficient Frontier via target-return sweeping.
        Returns list of (w, ret, vol, sharpe) for each frontier point.
        """
        mu  = self.mu_ann.values
        cov = self.cov_ann.values
        n   = self.n
        rf  = self.rf_ann

        ret_min = mu.min() * 1.01
        ret_max = mu.max() * 0.99
        targets = np.linspace(ret_min, ret_max, n_points)

        frontier = []
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
            ]
            bounds = [(0, 1)] * n
            w0     = np.ones(n) / n
            res    = minimize(lambda w: np.sqrt(w @ cov @ w), w0,
                              bounds=bounds, constraints=constraints, method="SLSQP")
            if res.success:
                w   = np.maximum(res.x, 0); w /= (w.sum() + 1e-12)
                vol = float(np.sqrt(w @ cov @ w))
                ret = float(w @ mu)
                frontier.append(FrontierPoint(
                    weights=w, ret=ret, vol=vol,
                    sharpe=(ret - rf) / (vol + 1e-12),
                    labels=self.tickers
                ))
        return frontier


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantum Portfolio Optimizer (QAOA via Qiskit)
# ─────────────────────────────────────────────────────────────────────────────

class QuantumOptimizer:
    """
    QUBO → QAOA portfolio selection.

    Problem: Select k assets from n to minimize risk and maximize return.
    Binary variables: x_i ∈ {0,1} (asset selected or not)

    QUBO Hamiltonian:
      H_Q = λ_r Σ_i μ_i x_i  -  λ_v Σ_{ij} σ_{ij} x_i x_j
            + λ_c (Σ_i x_i - k)²

    Mapped to Ising:
      x_i = (1 - z_i)/2,  z_i ∈ {-1, +1}

    QAOA Circuit depth p:
      |ψ(β,γ)⟩ = U_B(β_p) U_C(γ_p) ... U_B(β_1) U_C(γ_1) |+⟩^n

    References:
      Farhi, Goldstone, Gutmann (2014) arXiv:1411.4028
      Mugel et al. (2022) Quantum 6, 684
    """

    def __init__(self, returns: pd.DataFrame, n_select: int = 5,
                 p: int = 2, backend: str = "aer_simulator"):
        self.returns  = returns
        self.mu       = (returns.mean() * 252).values
        self.cov      = (returns.cov()  * 252).values
        self.n        = len(returns.columns)
        self.n_select = min(n_select, self.n)
        self.tickers  = returns.columns.tolist()
        self.p        = p
        self.backend  = backend
        self._result  = None

    def _build_qubo(self, lambda_risk: float = 1.0,
                    lambda_ret: float = 2.0,
                    lambda_card: float = 5.0) -> np.ndarray:
        """
        Build QUBO matrix Q such that:
          objective = x^T Q x  (x ∈ {0,1}^n)
        """
        n   = self.n
        k   = self.n_select
        Q   = np.zeros((n, n))

        # Risk term: λ_v Σ_{ij} σ_{ij} x_i x_j
        Q  -= lambda_ret * np.diag(self.mu)    # linear: subtract return on diag
        Q  += lambda_risk * self.cov           # quadratic: add covariance

        # Cardinality penalty: λ_c (Σx_i - k)^2 = λ_c [Σ x_i^2 + 2 Σ_{i<j} x_i x_j - 2k Σ x_i + k²]
        Q  += lambda_card * np.ones((n, n))     # cross terms: 2λ upper triangle
        Q  += lambda_card * np.diag(np.ones(n) * (1 - 2*k))  # diagonal: (1-2k)
        # constant k² absorbed into energy offset

        return Q

    def _qubo_to_ising(self, Q: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Transform QUBO to Ising: x_i = (1 - z_i)/2
        H_Ising = Σ_i h_i z_i + Σ_{ij} J_{ij} z_i z_j + const
        """
        n = Q.shape[0]
        J: Dict[Tuple, float] = {}
        h: Dict[int, float]   = {}

        for i in range(n):
            # Diagonal Q_{ii} → h_i
            h[i] = h.get(i, 0) - Q[i, i] / 2
            for j in range(i+1, n):
                J[(i, j)] = (Q[i, j] + Q[j, i]) / 4
                h[i] = h.get(i, 0) - (Q[i, j] + Q[j, i]) / 4
                h[j] = h.get(j, 0) - (Q[i, j] + Q[j, i]) / 4

        return h, J

    def _run_qaoa(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run QAOA simulation using Qiskit Aer.
        Returns: (best_binary_vector, objective_value)
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
            from qiskit_aer import AerSimulator
            from scipy.optimize import minimize as sp_minimize

            n    = self.n
            p    = self.p
            beta  = ParameterVector("β", p)
            gamma = ParameterVector("γ", p)

            def build_circuit(beta_vals, gamma_vals):
                qc = QuantumCircuit(n)
                # Initial state |+>^n
                qc.h(range(n))
                for layer in range(p):
                    # Problem unitary U_C(γ)
                    for i in range(n):
                        for j in range(i+1, n):
                            coeff = Q[i, j] + Q[j, i]
                            if abs(coeff) > 1e-8:
                                qc.rzz(2 * gamma_vals[layer] * coeff, i, j)
                        if abs(Q[i, i]) > 1e-8:
                            qc.rz(2 * gamma_vals[layer] * Q[i, i], i)
                    # Mixer U_B(β)
                    qc.rx(2 * beta_vals[layer], range(n))
                qc.measure_all()
                return qc

            def qaoa_cost(params):
                beta_v  = params[:p]
                gamma_v = params[p:]
                qc      = build_circuit(beta_v, gamma_v)

                sim     = AerSimulator(method="statevector")
                from qiskit import transpile
                t_qc    = transpile(qc, sim)
                job     = sim.run(t_qc, shots=1024)
                counts  = job.result().get_counts()

                # Compute expectation value
                energy = 0.0
                total  = sum(counts.values())
                for bitstr, count in counts.items():
                    x = np.array([int(b) for b in reversed(bitstr[:n])])
                    energy += count * float(x @ Q @ x)
                return energy / total

            # Classical optimization of QAOA parameters
            x0    = np.random.uniform(0, np.pi, 2 * p)
            res   = sp_minimize(qaoa_cost, x0, method="COBYLA",
                                options={"maxiter": 200, "rhobeg": 0.5})

            # Sample final distribution
            beta_opt  = res.x[:p]
            gamma_opt = res.x[p:]
            qc_final  = build_circuit(beta_opt, gamma_opt)
            sim       = AerSimulator(method="statevector")
            from qiskit import transpile
            t_qc      = transpile(qc_final, sim)
            job       = sim.run(t_qc, shots=4096)
            counts    = job.result().get_counts()

            # Best sample
            best_x    = min(counts, key=lambda b: float(
                np.array([int(c) for c in reversed(b[:n])]) @ Q @
                np.array([int(c) for c in reversed(b[:n])])
            ))
            x_opt     = np.array([int(c) for c in reversed(best_x[:n])])
            obj       = float(x_opt @ Q @ x_opt)
            return x_opt, obj, res.fun

        except ImportError as e:
            log.warning(f"Qiskit not available: {e}. Using classical QUBO solver.")
            return self._brute_force_qubo(Q)

    def _brute_force_qubo(self, Q: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Classical brute-force QUBO (exact for n ≤ 20)."""
        n    = self.n
        k    = self.n_select
        best = None; best_val = np.inf
        from itertools import combinations
        for indices in combinations(range(n), k):
            x = np.zeros(n)
            x[list(indices)] = 1.0
            val = float(x @ Q @ x)
            if val < best_val:
                best_val = val
                best     = x.copy()
        return best, best_val, best_val

    def optimize(self) -> OptimizationResult:
        """
        Full quantum optimization pipeline:
        1. Build QUBO
        2. Run QAOA (or brute-force fallback)
        3. Post-process: equal-weight selected assets
        4. Return OptimizationResult comparable to classical
        """
        log.info(f"Running quantum optimizer: {self.n} assets, p={self.p}")
        Q = self._build_qubo()

        try:
            x_opt, obj, qaoa_energy = self._run_qaoa(Q)
        except Exception as e:
            log.warning(f"QAOA failed: {e}. Using brute-force.")
            x_opt, obj, qaoa_energy = self._brute_force_qubo(Q)

        # Selected assets
        selected = np.where(x_opt > 0.5)[0]
        if len(selected) == 0:
            selected = np.argsort(self.mu)[-self.n_select:]

        # Equal-weight within selection (can be refined with classical MV)
        n_sel = len(selected)
        w_full = np.zeros(self.n)
        w_full[selected] = 1.0 / n_sel

        r = float(w_full @ self.mu)
        v = float(np.sqrt(w_full @ self.cov @ w_full))
        rf_ann = 0.07

        self._result = OptimizationResult(
            method       = "quantum_qaoa",
            weights      = {self.tickers[i]: round(w_full[i], 6) for i in range(self.n)},
            expected_ret = round(r, 4),
            expected_vol = round(v, 4),
            sharpe       = round((r - rf_ann) / (v + 1e-12), 4),
            metadata     = {
                "selected_assets": [self.tickers[i] for i in selected],
                "qubo_energy":      round(obj, 6),
                "qaoa_depth":       self.p,
                "n_selected":       n_sel,
            }
        )
        return self._result

    def quantum_frontier(self, n_points: int = 10) -> List[FrontierPoint]:
        """
        Approximate quantum frontier by varying λ_risk/λ_ret trade-off.
        """
        frontier = []
        for lam in np.linspace(0.2, 3.0, n_points):
            Q = self._build_qubo(lambda_risk=lam, lambda_ret=2.0)
            try:
                x, _, _ = self._brute_force_qubo(Q)
            except:
                continue
            selected = np.where(x > 0.5)[0]
            if len(selected) == 0:
                continue
            w = np.zeros(self.n)
            w[selected] = 1.0 / len(selected)
            r = float(w @ self.mu)
            v = float(np.sqrt(w @ self.cov @ w))
            frontier.append(FrontierPoint(
                weights=w, ret=r, vol=v,
                sharpe=(r - 0.07) / (v + 1e-12),
                labels=self.tickers
            ))
        return frontier


# ─────────────────────────────────────────────────────────────────────────────
# 3. Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def equal_weight_portfolio(tickers: List[str]) -> Dict[str, float]:
    n = len(tickers)
    return {t: 1/n for t in tickers}


def buy_and_hold_benchmark(returns: pd.DataFrame,
                           weights: Optional[np.ndarray] = None) -> pd.Series:
    """Cumulative return of buy-and-hold strategy."""
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    port_r = (returns * weights).sum(axis=1)
    return (1 + port_r).cumprod()


def compute_all_portfolios(
    returns: pd.DataFrame,
    n_quantum_assets: int = 10
) -> Dict[str, OptimizationResult]:
    """
    Compute Max Sharpe, Min Vol, Risk Parity, Quantum — single call.
    """
    classical = ClassicalOptimizer(returns)
    results   = {}

    results["max_sharpe"]   = classical.max_sharpe()
    results["min_vol"]      = classical.min_volatility()
    results["risk_parity"]  = classical.risk_parity()

    # Equal weight baseline
    n = len(returns.columns)
    ew_w = np.ones(n) / n
    mu   = (returns.mean() * 252).values
    cov  = (returns.cov()  * 252).values
    ew_r = float(ew_w @ mu)
    ew_v = float(np.sqrt(ew_w @ cov @ ew_w))
    results["equal_weight"] = OptimizationResult(
        method="equal_weight",
        weights=dict(zip(returns.columns, ew_w)),
        expected_ret=round(ew_r, 4),
        expected_vol=round(ew_v, 4),
        sharpe=round((ew_r - 0.07) / (ew_v + 1e-12), 4),
    )

    # Quantum (on subset for speed)
    q_returns = returns.iloc[:, :min(n_quantum_assets, n)]
    quantum   = QuantumOptimizer(q_returns, n_select=min(5, n_quantum_assets))
    results["quantum"]      = quantum.optimize()

    return results
