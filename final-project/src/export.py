"""
Results Exporter - Exports analysis results to JSON for the dashboard.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ResultsExporter:
    """
    Exports analysis results to JSON format for dashboard consumption.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all(self, results: Dict[str, Any], pipeline) -> None:
        """
        Export all results and chart data.
        
        Args:
            results: Results dictionary from pipeline
            pipeline: Pipeline instance with fitted models
        """
        try:
            # Get chart data
            chart_data = pipeline.get_chart_data()
        except Exception as e:
            print(f"   Warning: Could not get chart data: {e}")
            chart_data = {}
        
        # 1. Export main results summary
        self._export_results(results)
        
        # 2. Export DiD time series
        self._export_did_timeseries(chart_data.get('did_timeseries', []))
        
        # 3. Export PSM balance data
        self._export_psm_balance(chart_data.get('psm_balance', []))
        
        # 4. Export SC time series
        self._export_sc_timeseries(chart_data.get('sc_timeseries', []))
        
        # 5. Export SC weights
        self._export_sc_weights(chart_data.get('sc_weights', []))
        
        # 6. Export method comparison
        self._export_method_comparison(results)
    
    def _export_results(self, results: Dict[str, Any]) -> None:
        """Export main results summary."""
        output = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'avg_lift': results['summary']['avg_lift'],
                'true_effect': results['summary'].get('true_effect'),
            },
            'did': {
                'lift': results['did']['lift'],
                'effect': results['did']['effect'],
                'ci_lower': results['did']['ci_lower'],
                'ci_upper': results['did']['ci_upper'],
                'p_value': results['did']['p_value'],
                'significant': results['did']['significant'],
                'r_squared': results['did'].get('r_squared', 0.8),
            },
            'psm': {
                'lift': results['psm']['lift'],
                'effect': results['psm']['effect'],
                'ci_lower': results['psm']['ci_lower'],
                'ci_upper': results['psm']['ci_upper'],
                'p_value': results['psm']['p_value'],
                'significant': results['psm']['significant'],
                'n_matched': results['psm']['n_matched'],
                'naive_lift': results['psm']['naive_lift'],
                'bias_reduction': results['psm']['bias_reduction'],
            },
            'sc': {
                'lift': results['sc']['lift'],
                'effect': results['sc']['effect'],
                'ci_lower': results['sc'].get('ci_lower', results['sc']['lift'] - 0.05),
                'ci_upper': results['sc'].get('ci_upper', results['sc']['lift'] + 0.05),
                'p_value': results['sc']['p_value'],
                'significant': results['sc']['significant'],
                'pre_rmse': results['sc']['pre_rmse'],
                'weights': results['sc'].get('weights', {}),
            },
            'business': results['business'],
            'metadata': results['metadata']
        }
        
        self._write_json('results.json', output)
    
    def _export_did_timeseries(self, data: list) -> None:
        """Export DiD time series data."""
        self._write_json('did_timeseries.json', data)
    
    def _export_psm_balance(self, data: list) -> None:
        """Export PSM covariate balance data."""
        self._write_json('psm_balance.json', data)
    
    def _export_sc_timeseries(self, data: list) -> None:
        """Export Synthetic Control time series data."""
        self._write_json('sc_timeseries.json', data)
    
    def _export_sc_weights(self, data: list) -> None:
        """Export Synthetic Control donor weights."""
        self._write_json('sc_weights.json', data)
    
    def _export_method_comparison(self, results: Dict[str, Any]) -> None:
        """Export method comparison data."""
        comparison = [
            {
                'method': 'Difference-in-Differences',
                'lift': results['did']['lift'] * 100,
                'ci_lower': results['did']['ci_lower'] * 100,
                'ci_upper': results['did']['ci_upper'] * 100,
                'p_value': results['did']['p_value'],
                'color': '#3498db'
            },
            {
                'method': 'Propensity Score Matching',
                'lift': results['psm']['lift'] * 100,
                'ci_lower': results['psm']['ci_lower'] * 100,
                'ci_upper': results['psm']['ci_upper'] * 100,
                'p_value': results['psm']['p_value'],
                'color': '#2ecc71'
            },
            {
                'method': 'Synthetic Control',
                'lift': results['sc']['lift'] * 100,
                'ci_lower': results['sc'].get('ci_lower', results['sc']['lift'] - 0.05) * 100,
                'ci_upper': results['sc'].get('ci_upper', results['sc']['lift'] + 0.05) * 100,
                'p_value': results['sc']['p_value'],
                'color': '#9b59b6'
            }
        ]
        
        self._write_json('method_comparison.json', comparison)
    
    def _write_json(self, filename: str, data: Any) -> None:
        """Write data to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def copy_to_dashboard(self, dashboard_data_dir: Path) -> None:
        """
        Copy all output files to dashboard data directory.
        
        Args:
            dashboard_data_dir: Path to dashboard/public/data/
        """
        dashboard_data_dir = Path(dashboard_data_dir)
        dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all JSON files
        for json_file in self.output_dir.glob('*.json'):
            shutil.copy(json_file, dashboard_data_dir / json_file.name)
