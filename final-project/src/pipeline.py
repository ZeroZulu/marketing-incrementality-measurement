"""
Main Pipeline - Orchestrates the full incrementality analysis.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from data.data_generator import SyntheticDataGenerator
from models.did import DifferenceInDifferences
from models.psm import PropensityScoreMatching
from models.synthetic_control import SyntheticControl


class IncrementalityPipeline:
    """
    Main pipeline for running incrementality analysis.
    
    Orchestrates data loading, model execution, and results aggregation.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        data_source: str = "synthetic",
        verbose: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            data_source: "synthetic" or "bigquery"
            verbose: Whether to print detailed output
        """
        self.config_path = Path(config_path)
        self.data_source = data_source
        self.verbose = verbose
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize storage
        self.data: Optional[pd.DataFrame] = None
        self.timeseries_data: Optional[pd.DataFrame] = None
        self.region_data: Optional[pd.DataFrame] = None
        
        # Results storage
        self.did_results: Optional[Dict] = None
        self.psm_results: Optional[Dict] = None
        self.sc_results: Optional[Dict] = None
        
        # Model instances (for accessing fitted objects)
        self.did_model: Optional[DifferenceInDifferences] = None
        self.psm_model: Optional[PropensityScoreMatching] = None
        self.sc_model: Optional[SyntheticControl] = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from configured source.
        
        Returns:
            DataFrame with user-level data
        """
        if self.data_source == "synthetic":
            return self._load_synthetic_data()
        elif self.data_source == "bigquery":
            return self._load_bigquery_data()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def _load_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic data for demo/testing."""
        synthetic_config = self.config.get('data', {}).get('synthetic', {})
        treatment_config = self.config.get('treatment', {})
        
        generator = SyntheticDataGenerator(
            n_users=synthetic_config.get('n_users', 50000),
            n_regions=synthetic_config.get('n_regions', 10),
            treatment_regions=synthetic_config.get('treatment_regions', ['California', 'New York']),
            true_effect=treatment_config.get('true_effect', 0.15),
            confounding_strength=treatment_config.get('confounding_strength', 0.8),
            random_seed=synthetic_config.get('random_seed', 42)
        )
        
        # Generate all data types
        self.data = generator.generate_user_data()
        self.timeseries_data = generator.generate_timeseries_data()
        self.region_data = generator.generate_region_timeseries()
        
        # Store generator for later use
        self._generator = generator
        
        return self.data
    
    def _load_bigquery_data(self) -> pd.DataFrame:
        """Load data from BigQuery."""
        try:
            from data.bigquery_loader import BigQueryLoader
        except ImportError:
            raise ImportError(
                "BigQuery dependencies not installed. "
                "Run: pip install google-cloud-bigquery"
            )
        
        bq_config = self.config.get('data', {}).get('bigquery', {})
        
        loader = BigQueryLoader(
            project=bq_config.get('project'),
            dataset=bq_config.get('dataset'),
            credentials_path=bq_config.get('credentials_path')
        )
        
        self.data = loader.load_user_data()
        self.timeseries_data = loader.load_timeseries_data()
        self.region_data = loader.load_region_timeseries()
        
        return self.data
    
    def run_did(self) -> Dict[str, Any]:
        """
        Run Difference-in-Differences analysis.
        
        Returns:
            Dictionary with DiD results
        """
        if self.region_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        did_config = self.config.get('models', {}).get('did', {})
        treatment_regions = self.config.get('data', {}).get('synthetic', {}).get(
            'treatment_regions', ['California', 'New York']
        )
        
        self.did_model = DifferenceInDifferences(
            cluster_robust_se=did_config.get('cluster_robust_se', True)
        )
        
        self.did_results = self.did_model.fit(
            data=self.region_data,
            treatment_regions=treatment_regions,
            treatment_period=26  # Week 26 = start of H2
        )
        
        return self.did_results
    
    def run_psm(self) -> Dict[str, Any]:
        """
        Run Propensity Score Matching analysis.
        
        Returns:
            Dictionary with PSM results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        psm_config = self.config.get('models', {}).get('psm', {})
        
        self.psm_model = PropensityScoreMatching(
            covariates=psm_config.get('covariates', [
                'days_since_install', 'session_count', 'engagement_score',
                'lifetime_value', 'is_mobile', 'is_organic'
            ]),
            matching_method=psm_config.get('matching_method', 'nearest'),
            caliper=psm_config.get('caliper', 0.1)
        )
        
        self.psm_results = self.psm_model.fit(
            data=self.data,
            treatment_col='treated',
            outcome_col='converted'
        )
        
        return self.psm_results
    
    def run_synthetic_control(self) -> Dict[str, Any]:
        """
        Run Synthetic Control analysis.
        
        Returns:
            Dictionary with SC results
        """
        if self.region_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        sc_config = self.config.get('models', {}).get('synthetic_control', {})
        
        self.sc_model = SyntheticControl(
            treatment_unit="California",
            donor_units=sc_config.get('donor_regions', [
                'Texas', 'Florida', 'Illinois', 'Ohio', 'Pennsylvania'
            ]),
            treatment_period=26
        )
        
        self.sc_results = self.sc_model.fit(
            data=self.region_data,
            outcome_col='conversions',
            time_col='week',
            unit_col='region'
        )
        
        return self.sc_results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all results in a structured format.
        
        Returns:
            Dictionary containing all analysis results
        """
        if not all([self.did_results, self.psm_results, self.sc_results]):
            raise ValueError(
                "Not all analyses complete. Run all models first."
            )
        
        # Calculate average lift
        avg_lift = np.mean([
            self.did_results['lift'],
            self.psm_results['lift'],
            self.sc_results['lift']
        ])
        
        # Business impact
        business_config = self.config.get('business', {})
        campaign_spend = business_config.get('campaign_spend', 500000)
        revenue_per_install = business_config.get('revenue_per_install', 25)
        
        # Estimate incremental installs (simplified)
        base_installs = campaign_spend / 10  # Assume $10 CPI baseline
        incremental_installs = int(base_installs * avg_lift)
        incremental_revenue = incremental_installs * revenue_per_install
        iroas = incremental_revenue / campaign_spend
        
        # Get true effect if using synthetic data
        true_effect = self.config.get('treatment', {}).get('true_effect')
        
        return {
            'did': self.did_results,
            'psm': self.psm_results,
            'sc': self.sc_results,
            'summary': {
                'avg_lift': avg_lift,
                'true_effect': true_effect,
                'bias_vs_true': (avg_lift - true_effect) / true_effect if true_effect else None
            },
            'business': {
                'campaign_spend': campaign_spend,
                'revenue_per_install': revenue_per_install,
                'incremental_installs': incremental_installs,
                'incremental_revenue': incremental_revenue,
                'iroas': iroas
            },
            'metadata': {
                'data_source': self.data_source,
                'n_users': len(self.data) if self.data is not None else 0,
                'config_path': str(self.config_path)
            }
        }
    
    def get_chart_data(self) -> Dict[str, Any]:
        """
        Get data formatted for dashboard charts.
        
        Returns:
            Dictionary with chart-ready data
        """
        chart_data = {}
        
        # DiD timeseries
        if self.did_model is not None:
            try:
                chart_data['did_timeseries'] = self.did_model.get_timeseries_data()
            except Exception as e:
                print(f"   Warning: Could not get DiD timeseries: {e}")
                chart_data['did_timeseries'] = []
        
        # PSM balance
        if self.psm_model is not None:
            try:
                chart_data['psm_balance'] = self.psm_model.get_balance_data()
            except Exception as e:
                print(f"   Warning: Could not get PSM balance: {e}")
                chart_data['psm_balance'] = []
            try:
                chart_data['psm_propensity'] = self.psm_model.get_propensity_data()
            except Exception as e:
                print(f"   Warning: Could not get PSM propensity: {e}")
                chart_data['psm_propensity'] = {'treated': [], 'control': []}
        
        # SC timeseries
        if self.sc_model is not None:
            try:
                chart_data['sc_timeseries'] = self.sc_model.get_timeseries_data()
            except Exception as e:
                print(f"   Warning: Could not get SC timeseries: {e}")
                chart_data['sc_timeseries'] = []
            try:
                chart_data['sc_weights'] = self.sc_model.get_weights_data()
            except Exception as e:
                print(f"   Warning: Could not get SC weights: {e}")
                chart_data['sc_weights'] = []
        
        return chart_data
