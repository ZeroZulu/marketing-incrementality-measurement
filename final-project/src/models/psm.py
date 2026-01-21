"""
Propensity Score Matching (PSM) for Causal Inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class PropensityScoreMatching:
    """Propensity Score Matching estimator."""
    
    def __init__(
        self,
        covariates: List[str] = None,
        matching_method: str = 'nearest',
        caliper: float = 0.1,
        n_neighbors: int = 1
    ):
        self.covariates = covariates or [
            'days_since_install', 'session_count', 'engagement_score',
            'lifetime_value', 'is_mobile', 'is_organic'
        ]
        self.matching_method = matching_method
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        
        self.propensity_model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self._balance_data: Optional[List[Dict]] = None
        self._propensity_scores: Optional[np.ndarray] = None
        self._treatment: Optional[np.ndarray] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        treatment_col: str = 'treated',
        outcome_col: str = 'converted'
    ) -> Dict[str, Any]:
        df = data.copy().reset_index(drop=True)
        
        available_covariates = [c for c in self.covariates if c in df.columns]
        if len(available_covariates) == 0:
            raise ValueError("No matching covariates found in data")
        
        X = df[available_covariates].values
        treatment = df[treatment_col].values.astype(int)
        outcome = df[outcome_col].values
        
        self._treatment = treatment
        
        # Scale and fit propensity model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        self.propensity_model.fit(X_scaled, treatment)
        propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        self._propensity_scores = propensity_scores
        
        # Balance before matching
        balance_before = self._calculate_balance(df, available_covariates, treatment)
        
        # Perform matching
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        if len(control_idx) == 0:
            raise ValueError("No control units found")
        
        # Match each treated to nearest control
        treated_ps = propensity_scores[treated_idx].reshape(-1, 1)
        control_ps = propensity_scores[control_idx].reshape(-1, 1)
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(control_ps)
        distances, indices = nn.kneighbors(treated_ps)
        
        # Get matched pairs
        matched_treated_idx = treated_idx
        matched_control_idx = control_idx[indices.flatten()]
        
        # Apply caliper if needed
        if self.matching_method == 'caliper':
            valid = distances.flatten() <= self.caliper
            matched_treated_idx = matched_treated_idx[valid]
            matched_control_idx = matched_control_idx[valid]
        
        # Calculate balance after matching
        if len(matched_treated_idx) > 0:
            matched_df = pd.concat([
                df.iloc[matched_treated_idx].assign(_grp=1),
                df.iloc[matched_control_idx].assign(_grp=0)
            ], ignore_index=True)
            balance_after = self._calculate_balance(matched_df, available_covariates, matched_df['_grp'].values)
        else:
            balance_after = {c: 0 for c in available_covariates}
        
        self._balance_data = self._create_balance_comparison(available_covariates, balance_before, balance_after)
        
        # Calculate ATT
        treated_outcomes = outcome[matched_treated_idx]
        control_outcomes = outcome[matched_control_idx]
        
        att = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
        # Standard error
        se = np.sqrt(
            np.var(treated_outcomes) / max(len(treated_outcomes), 1) +
            np.var(control_outcomes) / max(len(control_outcomes), 1)
        )
        
        # Lift calculation
        control_mean = np.mean(control_outcomes)
        if control_mean == 0:
            control_mean = 0.001
        
        lift = att / control_mean
        lift_se = se / abs(control_mean)
        
        # Clip extreme values
        lift = np.clip(lift, -1.0, 1.0)
        
        ci_lower = lift - 1.96 * lift_se
        ci_upper = lift + 1.96 * lift_se
        
        # P-value
        if len(treated_outcomes) > 1 and len(control_outcomes) > 1:
            _, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
        else:
            p_value = 1.0
        
        # Naive estimate
        naive_t = np.mean(outcome[treatment == 1])
        naive_c = np.mean(outcome[treatment == 0])
        naive_c = max(naive_c, 0.001)
        naive_lift = np.clip((naive_t - naive_c) / naive_c, -1.0, 1.0)
        
        return {
            'method': 'Propensity Score Matching',
            'effect': att,
            'lift': lift,
            'se': lift_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': self._format_pvalue(p_value),
            'significant': p_value < 0.05,
            'n_matched': len(matched_treated_idx),
            'n_treated': int(np.sum(treatment)),
            'n_control': int(np.sum(1 - treatment)),
            'naive_lift': naive_lift,
            'bias_reduction': (naive_lift - lift) / naive_lift if naive_lift != 0 else 0,
            'mean_propensity_treated': float(np.mean(propensity_scores[treatment == 1])),
            'mean_propensity_control': float(np.mean(propensity_scores[treatment == 0]))
        }
    
    def _calculate_balance(self, df: pd.DataFrame, covariates: List[str], treatment: np.ndarray) -> Dict[str, float]:
        balance = {}
        for cov in covariates:
            t_vals = df.loc[treatment == 1, cov]
            c_vals = df.loc[treatment == 0, cov]
            if len(t_vals) == 0 or len(c_vals) == 0:
                balance[cov] = 0
                continue
            pooled_std = np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2)
            if pooled_std > 0:
                balance[cov] = (t_vals.mean() - c_vals.mean()) / pooled_std
            else:
                balance[cov] = 0
        return balance
    
    def _create_balance_comparison(self, covariates: List[str], before: Dict, after: Dict) -> List[Dict]:
        return [
            {'covariate': c.replace('_', ' ').title(), 'before': round(before.get(c, 0), 3), 'after': round(after.get(c, 0), 3)}
            for c in covariates
        ]
    
    def _format_pvalue(self, p: float) -> str:
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"
    
    def get_balance_data(self) -> List[Dict]:
        return self._balance_data or []
    
    def get_propensity_data(self) -> Dict[str, List[float]]:
        if self._propensity_scores is None or self._treatment is None:
            return {'treated': [], 'control': []}
        try:
            return {
                'treated': self._propensity_scores[self._treatment == 1][:500].tolist(),
                'control': self._propensity_scores[self._treatment == 0][:500].tolist()
            }
        except:
            return {'treated': [], 'control': []}
