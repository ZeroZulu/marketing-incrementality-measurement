"""
Synthetic Control Method for Causal Inference.

Constructs a weighted combination of control units to create a 
"synthetic" version of the treated unit for counterfactual estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy.optimize import minimize


class SyntheticControl:
    """
    Synthetic Control Method estimator.
    
    Creates a synthetic control unit as a weighted average of donor pool
    units that best matches the treated unit's pre-treatment trajectory.
    
    Key assumption: Good pre-treatment fit (synthetic closely matches 
    treated unit before intervention)
    """
    
    def __init__(
        self,
        treatment_unit: str,
        donor_units: List[str],
        treatment_period: int
    ):
        """
        Initialize Synthetic Control model.
        
        Args:
            treatment_unit: Name of the treated unit
            donor_units: List of control unit names (donor pool)
            treatment_period: Time period when treatment started
        """
        self.treatment_unit = treatment_unit
        self.donor_units = donor_units
        self.treatment_period = treatment_period
        
        # Results storage
        self.weights: Optional[np.ndarray] = None
        self.weights_dict: Optional[Dict[str, float]] = None
        self._timeseries_data: Optional[pd.DataFrame] = None
        self.pre_rmse: Optional[float] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        outcome_col: str = 'conversions',
        time_col: str = 'week',
        unit_col: str = 'region'
    ) -> Dict[str, Any]:
        """
        Fit the Synthetic Control model.
        
        Args:
            data: Long-format DataFrame with time series data
            outcome_col: Name of outcome column
            time_col: Name of time column
            unit_col: Name of unit identifier column
            
        Returns:
            Dictionary with estimation results
        """
        # Pivot to wide format (time x units)
        pivot = data.pivot_table(
            index=time_col,
            columns=unit_col,
            values=outcome_col,
            aggfunc='mean'
        ).reset_index()
        
        # Check that treatment unit and donors exist
        if self.treatment_unit not in pivot.columns:
            raise ValueError(f"Treatment unit '{self.treatment_unit}' not found in data")
        
        available_donors = [d for d in self.donor_units if d in pivot.columns]
        if len(available_donors) == 0:
            raise ValueError("No donor units found in data")
        
        # Extract pre-treatment data
        pre_mask = pivot[time_col] < self.treatment_period
        post_mask = pivot[time_col] >= self.treatment_period
        
        pre_treated = pivot.loc[pre_mask, self.treatment_unit].values
        pre_donors = pivot.loc[pre_mask, available_donors].values
        
        # Optimize weights to minimize pre-treatment RMSE
        weights = self._optimize_weights(pre_treated, pre_donors)
        
        # Store weights
        self.weights = weights
        self.weights_dict = dict(zip(available_donors, weights))
        
        # Calculate synthetic control for all periods
        all_donors = pivot[available_donors].values
        synthetic = all_donors @ weights
        
        # Get actual treated values
        actual = pivot[self.treatment_unit].values
        weeks = pivot[time_col].values
        
        # Calculate effect
        post_actual = actual[post_mask]
        post_synthetic = synthetic[post_mask]
        
        # Average treatment effect
        effect = np.mean(post_actual - post_synthetic)
        
        # Pre-treatment fit (RMSE)
        pre_synthetic = synthetic[pre_mask]
        self.pre_rmse = np.sqrt(np.mean((pre_treated - pre_synthetic) ** 2))
        
        # Calculate lift
        baseline = np.mean(pre_treated)
        lift = effect / baseline if baseline > 0 else 0
        
        # Standard error via placebo (simplified)
        # Run synthetic control on each donor unit as if it were treated
        placebo_effects = []
        for donor in available_donors:
            try:
                placebo_effect = self._run_placebo(pivot, donor, available_donors, time_col)
                placebo_effects.append(placebo_effect)
            except:
                continue
        
        if len(placebo_effects) > 0:
            se = np.std(placebo_effects)
            # P-value: proportion of placebos with effect >= actual
            p_value = np.mean(np.abs(placebo_effects) >= abs(effect))
        else:
            se = 0.05
            p_value = 0.05
        
        # Confidence interval (placebo-based)
        ci_lower = lift - 1.96 * (se / baseline) if baseline > 0 else lift - 0.1
        ci_upper = lift + 1.96 * (se / baseline) if baseline > 0 else lift + 0.1
        
        # Store timeseries data for charts
        self._timeseries_data = pd.DataFrame({
            'week': weeks,
            'actual': actual,
            'synthetic': synthetic,
            'gap': actual - synthetic,
            'period': ['Post' if w >= self.treatment_period else 'Pre' for w in weeks]
        })
        
        return {
            'method': 'Synthetic Control',
            'effect': effect,
            'lift': lift,
            'se': se / baseline if baseline > 0 else se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': self._format_pvalue(p_value),
            'significant': p_value < 0.05,
            'pre_rmse': self.pre_rmse,
            'weights': self.weights_dict,
            'n_pre_periods': int(np.sum(pre_mask)),
            'n_post_periods': int(np.sum(post_mask)),
            'n_donors': len(available_donors)
        }
    
    def _optimize_weights(
        self,
        treated: np.ndarray,
        donors: np.ndarray
    ) -> np.ndarray:
        """
        Optimize donor weights to minimize pre-treatment RMSE.
        
        Constraints:
        - Weights sum to 1
        - All weights >= 0
        """
        n_donors = donors.shape[1]
        
        # Objective: minimize squared distance
        def objective(w):
            synthetic = donors @ w
            return np.sum((treated - synthetic) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds: 0 <= w <= 1
        bounds = [(0, 1) for _ in range(n_donors)]
        
        # Initial guess: equal weights
        w0 = np.ones(n_donors) / n_donors
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x
    
    def _run_placebo(
        self,
        pivot: pd.DataFrame,
        placebo_unit: str,
        donor_units: List[str],
        time_col: str
    ) -> float:
        """
        Run placebo test: treat a control unit as if it were treated.
        
        Returns:
            Placebo treatment effect
        """
        # Remove placebo unit from donors
        placebo_donors = [d for d in donor_units if d != placebo_unit]
        
        if len(placebo_donors) < 2:
            return 0
        
        pre_mask = pivot[time_col] < self.treatment_period
        post_mask = pivot[time_col] >= self.treatment_period
        
        pre_placebo = pivot.loc[pre_mask, placebo_unit].values
        pre_donors = pivot.loc[pre_mask, placebo_donors].values
        
        # Optimize weights
        weights = self._optimize_weights(pre_placebo, pre_donors)
        
        # Calculate synthetic for post period
        post_donors = pivot.loc[post_mask, placebo_donors].values
        post_synthetic = post_donors @ weights
        post_actual = pivot.loc[post_mask, placebo_unit].values
        
        return np.mean(post_actual - post_synthetic)
    
    def _format_pvalue(self, p: float) -> str:
        """Format p-value for display."""
        if p < 0.001:
            return "< 0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        else:
            return f"{p:.3f}"
    
    def get_timeseries_data(self) -> List[Dict]:
        """
        Get time series data for charts.
        
        Returns:
            List of dictionaries with week, actual, synthetic, gap values
        """
        if self._timeseries_data is None:
            return []
        
        return self._timeseries_data.round(2).to_dict('records')
    
    def get_weights_data(self) -> List[Dict]:
        """
        Get donor weights for charts.
        
        Returns:
            List of dictionaries with state and weight values
        """
        if self.weights_dict is None:
            return []
        
        return [
            {'state': state, 'weight': round(weight, 3)}
            for state, weight in sorted(
                self.weights_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
