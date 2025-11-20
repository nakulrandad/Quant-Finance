"""Portfolio Optimization Strategies"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """Calculate optimal portfolio weights.

        Args:
            returns: DataFrame of asset returns
            **kwargs: Strategy-specific parameters

        Returns:
            pd.Series: Optimal weights for each asset
        """
        pass

    def _get_default_constraints(self, n_assets):
        """Get default optimization constraints (sum of weights = 1, 0 <= weight <= 1)."""
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        return bounds, constraints


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Mean-Variance Optimization Strategy."""

    def optimize(  # type: ignore[override]
        self,
        returns: pd.DataFrame,
        mu=None,
        sigma=None,
        bounds="default",
        constraints="default",
    ) -> pd.Series:
        """Calculate weights that maximize the Sharpe ratio.

        Args:
            returns: DataFrame of asset returns (pre-aggregated if needed)
            mu: Expected returns (uses historical means if None)
            sigma: Covariance matrix (uses historical covariance if None)
            bounds: Optimization bounds
            constraints: Optimization constraints

        Returns:
            pd.Series: Optimal weights
        """
        if mu is None:
            mu = returns.mean()

        if sigma is None:
            sigma = returns.cov()

        n_assets = len(returns.columns)
        assets = returns.columns

        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * mu)
            portfolio_vol = np.sqrt(weights.T @ sigma @ weights)
            if portfolio_vol == 0:
                return 1e6
            return -(portfolio_return / portfolio_vol)

        if bounds == "default" or constraints == "default":
            default_bounds, default_constraints = self._get_default_constraints(
                n_assets
            )
            bounds = default_bounds if bounds == "default" else bounds
            constraints = (
                default_constraints if constraints == "default" else constraints
            )

        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=assets)


class KellyOptimizer(PortfolioOptimizer):
    """Kelly Criterion Optimization Strategy."""

    def optimize(  # type: ignore[override]
        self,
        returns: pd.DataFrame,
        bounds="default",
        constraints="default",
    ) -> pd.Series:
        """Calculate weights that maximize log returns.

        Args:
            returns: DataFrame of asset returns (pre-aggregated if needed)
            bounds: Optimization bounds
            constraints: Optimization constraints

        Returns:
            pd.Series: Optimal weights
        """
        n_assets = len(returns.columns)
        assets = returns.columns

        def neg_log_return(weights):
            portfolio_returns = returns.dot(weights)
            return -np.mean(np.log(1 + portfolio_returns))

        if bounds == "default" or constraints == "default":
            default_bounds, default_constraints = self._get_default_constraints(
                n_assets
            )
            bounds = default_bounds if bounds == "default" else bounds
            constraints = (
                default_constraints if constraints == "default" else constraints
            )

        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_log_return,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        return pd.Series(result.x, index=assets)


class RiskBudgetOptimizer(PortfolioOptimizer):
    """Risk Budgeting Optimization Strategy."""

    def optimize(  # type: ignore[override]
        self,
        returns: pd.DataFrame,
        risk_budget=None,
    ) -> pd.Series:
        """Calculate weights that match a target risk budget.

        Args:
            returns: DataFrame of asset returns (pre-aggregated if needed)
            risk_budget: Target risk contribution for each asset (sums to 1)

        Returns:
            pd.Series: Optimal weights
        """
        n_assets = len(returns.columns)
        assets = returns.columns

        if risk_budget is None:
            risk_budget = np.array([1 / n_assets] * n_assets)
        else:
            risk_budget = np.array(risk_budget)
            if not np.isclose(np.sum(risk_budget), 1.0, atol=1e-6):
                raise ValueError("Risk budget must sum to 1")
            if len(risk_budget) != n_assets:
                raise ValueError(
                    f"Risk budget length ({len(risk_budget)}) must match number of assets ({n_assets})"
                )

        sigma = returns.cov()

        def risk_budget_objective(weights):
            # Normalize weights to sum to 1 for the calculation
            # Note: We enforce sum=1 constraint in optimization anyway
            weights = weights / np.sum(weights)
            portfolio_var = weights.T @ sigma @ weights
            risk_contributions = (sigma @ weights) * weights / portfolio_var
            return np.sum((risk_contributions - risk_budget) ** 2)

        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        initial_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        if not result.success:
            raise ValueError(f"Risk budget optimization failed: {result.message}")

        return pd.Series(result.x, index=assets)
