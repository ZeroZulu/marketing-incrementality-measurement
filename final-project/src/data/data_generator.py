"""
Synthetic Data Generator for Marketing Incrementality Analysis.

Generates realistic user-level and time-series data with known treatment effects.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class SyntheticDataGenerator:
    """
    Generates synthetic data with known treatment effects.
    """
    
    def __init__(
        self,
        n_users: int = 50000,
        n_regions: int = 10,
        treatment_regions: List[str] = None,
        true_effect: float = 0.15,
        confounding_strength: float = 0.3,
        random_seed: int = 42,
        verbose: bool = False
    ):
        self.n_users = n_users
        self.n_regions = n_regions
        self.treatment_regions = treatment_regions or ['California', 'New York']
        self.true_effect = true_effect
        self.confounding_strength = confounding_strength
        self.random_seed = random_seed
        self.verbose = verbose
        
        np.random.seed(random_seed)
        
        self.all_regions = [
            'California', 'New York', 'Texas', 'Florida', 'Illinois',
            'Ohio', 'Pennsylvania', 'Georgia', 'Michigan', 'Arizona'
        ][:n_regions]
        
        self.control_regions = [r for r in self.all_regions if r not in self.treatment_regions]
    
    def generate_user_data(self) -> pd.DataFrame:
        """Generate user-level data with covariates and outcomes."""
        
        user_ids = [f"user_{i:06d}" for i in range(self.n_users)]
        
        # Assign regions
        region_weights = np.array([0.15, 0.12, 0.12, 0.10, 0.08, 
                                   0.08, 0.07, 0.06, 0.06, 0.06][:self.n_regions])
        region_weights = region_weights / region_weights.sum()
        regions = np.random.choice(self.all_regions, size=self.n_users, p=region_weights)
        
        # Generate covariates FIRST (before treatment)
        days_since_install = np.random.exponential(90, self.n_users)
        days_since_install = np.clip(days_since_install, 1, 365).astype(int)
        
        session_count = np.random.poisson(15, self.n_users)
        session_count = np.clip(session_count, 0, 100).astype(int)
        
        engagement_score = np.random.beta(2, 5, self.n_users) * 100
        engagement_score = np.clip(engagement_score, 0, 100)
        
        lifetime_value = np.random.exponential(20, self.n_users)
        lifetime_value = np.clip(lifetime_value, 0, 500)
        
        is_mobile = np.random.binomial(1, 0.7, self.n_users)
        is_organic = np.random.binomial(1, 0.4, self.n_users)
        
        # Treatment assignment with selection bias
        # Users with higher engagement MORE likely to be treated (selection bias)
        treatment_propensity = (
            0.3 +  # Base probability
            self.confounding_strength * 0.3 * (engagement_score / 100) +
            self.confounding_strength * 0.2 * (session_count / 50) +
            0.1 * np.array([r in self.treatment_regions for r in regions])  # Region effect
        )
        treatment_propensity = np.clip(treatment_propensity, 0.1, 0.9)
        treated = np.random.binomial(1, treatment_propensity)
        
        # Generate outcome (conversion)
        # Base probability depends on covariates
        base_prob = (
            0.08 +
            0.05 * (engagement_score / 100) +
            0.03 * (session_count / 50) +
            0.02 * np.log1p(lifetime_value) / 5 +
            0.02 * is_mobile -
            0.01 * is_organic
        )
        base_prob = np.clip(base_prob, 0.02, 0.5)
        
        # Add TRUE treatment effect (only for treated users)
        conversion_prob = base_prob + treated * self.true_effect * base_prob
        conversion_prob = np.clip(conversion_prob, 0, 0.95)
        
        converted = np.random.binomial(1, conversion_prob)
        conversion_value = converted * np.random.exponential(50, self.n_users)
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'region': regions,
            'treated': treated,
            'days_since_install': days_since_install,
            'session_count': session_count,
            'engagement_score': engagement_score.round(2),
            'lifetime_value': lifetime_value.round(2),
            'is_mobile': is_mobile,
            'is_organic': is_organic,
            'converted': converted,
            'conversion_value': conversion_value.round(2)
        })
        
        if self.verbose:
            print(f"   Generated {len(df)} users")
            print(f"   Treatment rate: {df['treated'].mean():.1%}")
            print(f"   Conversion rate (treated): {df[df['treated']==1]['converted'].mean():.1%}")
            print(f"   Conversion rate (control): {df[df['treated']==0]['converted'].mean():.1%}")
        
        return df
    
    def generate_timeseries_data(self, n_weeks: int = 52) -> pd.DataFrame:
        """Generate weekly time-series data for DiD analysis."""
        treatment_week = n_weeks // 2
        records = []
        
        for region in self.all_regions:
            is_treated = region in self.treatment_regions
            base = 1000 + np.random.normal(0, 50)
            trend = np.linspace(0, 100, n_weeks)
            seasonality = 50 * np.sin(np.linspace(0, 4 * np.pi, n_weeks))
            random_walk = np.cumsum(np.random.normal(0, 10, n_weeks))
            
            for week in range(n_weeks):
                post = week >= treatment_week
                conversions = base + trend[week] + seasonality[week] + random_walk[week] + np.random.normal(0, 20)
                
                if is_treated and post:
                    conversions *= (1 + self.true_effect)
                
                records.append({
                    'region': region,
                    'week': week,
                    'post_treatment': int(post),
                    'treated': int(is_treated),
                    'conversions': max(0, int(conversions)),
                    'users': int(np.random.poisson(5000)),
                    'spend': np.random.exponential(10000) if is_treated and post else 0
                })
        
        return pd.DataFrame(records)
    
    def generate_region_timeseries(self, n_weeks: int = 52) -> pd.DataFrame:
        """Generate region-level time series for Synthetic Control."""
        treatment_week = n_weeks // 2
        common_trend = np.cumsum(np.random.normal(2, 5, n_weeks))
        common_seasonal = 100 * np.sin(np.linspace(0, 4 * np.pi, n_weeks))
        
        data = {'week': list(range(n_weeks))}
        
        for region in self.all_regions:
            is_treated = region in self.treatment_regions
            base = 1000 + np.random.normal(0, 100)
            region_trend = np.random.normal(0.5, 0.2)
            region_seasonal = np.random.normal(1, 0.2)
            
            series = []
            for week in range(n_weeks):
                value = (
                    base +
                    common_trend[week] * region_trend +
                    common_seasonal[week] * region_seasonal +
                    np.random.normal(0, 30)
                )
                
                if is_treated and week >= treatment_week:
                    value *= (1 + self.true_effect)
                
                series.append(max(0, value))
            
            data[region] = series
        
        df = pd.DataFrame(data)
        
        df_long = df.melt(id_vars=['week'], var_name='region', value_name='conversions')
        df_long['treated'] = df_long['region'].isin(self.treatment_regions).astype(int)
        df_long['post_treatment'] = (df_long['week'] >= treatment_week).astype(int)
        
        return df_long
    
    def get_true_effect(self) -> float:
        return self.true_effect
    
    def get_summary(self) -> dict:
        return {
            'n_users': self.n_users,
            'n_regions': self.n_regions,
            'treatment_regions': self.treatment_regions,
            'control_regions': self.control_regions,
            'true_effect': self.true_effect,
            'confounding_strength': self.confounding_strength,
            'random_seed': self.random_seed
        }
