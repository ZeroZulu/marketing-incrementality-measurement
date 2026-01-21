"""
Tests for Marketing Incrementality Models.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_generator import SyntheticDataGenerator
from models.did import DifferenceInDifferences
from models.psm import PropensityScoreMatching
from models.synthetic_control import SyntheticControl


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_generates_correct_number_of_users(self):
        gen = SyntheticDataGenerator(n_users=1000, random_seed=42)
        data = gen.generate_user_data()
        assert len(data) == 1000
    
    def test_has_required_columns(self):
        gen = SyntheticDataGenerator(n_users=100, random_seed=42)
        data = gen.generate_user_data()
        required = ['user_id', 'region', 'treated', 'converted']
        for col in required:
            assert col in data.columns
    
    def test_treatment_assignment(self):
        gen = SyntheticDataGenerator(
            n_users=1000,
            treatment_regions=['California'],
            random_seed=42
        )
        data = gen.generate_user_data()
        ca_data = data[data['region'] == 'California']
        assert all(ca_data['treated'] == 1)


class TestDifferenceInDifferences:
    """Test DiD model."""
    
    def test_fit_returns_results(self):
        gen = SyntheticDataGenerator(n_users=1000, random_seed=42)
        gen.generate_user_data()
        region_data = gen.generate_region_timeseries()
        
        model = DifferenceInDifferences()
        results = model.fit(
            data=region_data,
            treatment_regions=['California', 'New York'],
            treatment_period=26
        )
        
        assert 'lift' in results
        assert 'p_value' in results
        assert 'significant' in results
    
    def test_positive_lift_with_treatment_effect(self):
        gen = SyntheticDataGenerator(
            n_users=5000,
            true_effect=0.20,
            random_seed=42
        )
        gen.generate_user_data()
        region_data = gen.generate_region_timeseries()
        
        model = DifferenceInDifferences()
        results = model.fit(
            data=region_data,
            treatment_regions=['California', 'New York'],
            treatment_period=26
        )
        
        assert results['lift'] > 0


class TestPropensityScoreMatching:
    """Test PSM model."""
    
    def test_fit_returns_results(self):
        gen = SyntheticDataGenerator(n_users=1000, random_seed=42)
        data = gen.generate_user_data()
        
        model = PropensityScoreMatching(
            covariates=['days_since_install', 'session_count', 'engagement_score']
        )
        results = model.fit(data, treatment_col='treated', outcome_col='converted')
        
        assert 'lift' in results
        assert 'n_matched' in results
        assert results['n_matched'] > 0


class TestSyntheticControl:
    """Test Synthetic Control model."""
    
    def test_fit_returns_results(self):
        gen = SyntheticDataGenerator(n_users=1000, random_seed=42)
        gen.generate_user_data()
        region_data = gen.generate_region_timeseries()
        
        model = SyntheticControl(
            treatment_unit='California',
            donor_units=['Texas', 'Florida', 'Illinois'],
            treatment_period=26
        )
        results = model.fit(region_data)
        
        assert 'lift' in results
        assert 'weights' in results
        assert 'pre_rmse' in results
    
    def test_weights_sum_to_one(self):
        gen = SyntheticDataGenerator(n_users=1000, random_seed=42)
        gen.generate_user_data()
        region_data = gen.generate_region_timeseries()
        
        model = SyntheticControl(
            treatment_unit='California',
            donor_units=['Texas', 'Florida', 'Illinois'],
            treatment_period=26
        )
        results = model.fit(region_data)
        
        weight_sum = sum(results['weights'].values())
        assert abs(weight_sum - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
