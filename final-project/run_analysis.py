#!/usr/bin/env python3
"""
Marketing Incrementality Analysis - Main Entry Point

Run this script to execute the full analysis pipeline and export results
for the dashboard.

Usage:
    python run_analysis.py              # Use synthetic data (demo)
    python run_analysis.py --bigquery   # Use real BigQuery data
    python run_analysis.py --config custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import IncrementalityPipeline
from export import ResultsExporter


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         Marketing Incrementality & Lift Measurement               ║
    ║                                                                   ║
    ║         DiD • PSM • Synthetic Control                             ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results: dict):
    """Print results summary table."""
    print("\n" + "=" * 65)
    print("                        ANALYSIS SUMMARY")
    print("=" * 65)
    
    print(f"""
    ┌─────────────────────────┬─────────┬────────────────┬─────────┐
    │ Method                  │ Lift    │ 95% CI         │ p-value │
    ├─────────────────────────┼─────────┼────────────────┼─────────┤
    │ Difference-in-Diff      │ {results['did']['lift']*100:5.1f}%  │ [{results['did']['ci_lower']*100:.1f}%, {results['did']['ci_upper']*100:.1f}%]    │ {results['did']['p_value']:<7} │
    │ Propensity Score Match  │ {results['psm']['lift']*100:5.1f}%  │ [{results['psm']['ci_lower']*100:.1f}%, {results['psm']['ci_upper']*100:.1f}%]     │ {results['psm']['p_value']:<7} │
    │ Synthetic Control       │ {results['sc']['lift']*100:5.1f}%  │ Placebo-based  │ {results['sc']['p_value']:<7} │
    ├─────────────────────────┼─────────┼────────────────┼─────────┤
    │ Average (Triangulated)  │ {results['summary']['avg_lift']*100:5.1f}%  │ —              │ —       │
    └─────────────────────────┴─────────┴────────────────┴─────────┘
    """)
    
    print(f"""
    Business Impact:
    • Campaign Spend:      ${results['business']['campaign_spend']:,.0f}
    • Incremental Installs: {results['business']['incremental_installs']:,.0f}
    • Incremental Revenue:  ${results['business']['incremental_revenue']:,.0f}
    • iROAS:               {results['business']['iroas']:.2f}x
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Run Marketing Incrementality Analysis"
    )
    parser.add_argument(
        "--bigquery",
        action="store_true",
        help="Use BigQuery as data source (requires credentials)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Determine data source
    data_source = "bigquery" if args.bigquery else "synthetic"
    print(f"📊 Data source: {data_source.upper()}")
    print(f"📁 Config: {args.config}")
    print(f"📂 Output: {args.output}/")
    print()
    
    try:
        # Initialize pipeline
        print("🔄 Initializing pipeline...")
        pipeline = IncrementalityPipeline(
            config_path=args.config,
            data_source=data_source,
            verbose=args.verbose
        )
        
        # Run analysis
        print("🔄 Loading data...")
        pipeline.load_data()
        print(f"   ✅ Loaded {len(pipeline.data):,} records")
        
        print("🔄 Running Difference-in-Differences...")
        did_results = pipeline.run_did()
        print(f"   ✅ DiD complete: {did_results['lift']*100:.1f}% lift")
        
        print("🔄 Running Propensity Score Matching...")
        psm_results = pipeline.run_psm()
        print(f"   ✅ PSM complete: {psm_results['lift']*100:.1f}% lift")
        
        print("🔄 Running Synthetic Control...")
        sc_results = pipeline.run_synthetic_control()
        print(f"   ✅ SC complete: {sc_results['lift']*100:.1f}% lift")
        
        # Get all results
        results = pipeline.get_results()
        
        # Export results
        print(f"🔄 Exporting results to {args.output}/...")
        exporter = ResultsExporter(output_dir=args.output)
        exporter.export_all(results, pipeline)
        print("   ✅ Results exported")
        
        # Copy to dashboard
        dashboard_data_dir = Path("dashboard/public/data")
        if dashboard_data_dir.exists() or dashboard_data_dir.parent.exists():
            exporter.copy_to_dashboard(dashboard_data_dir)
            print(f"   ✅ Copied to dashboard/public/data/")
        
        # Print summary
        print_summary(results)
        
        print("\n" + "=" * 65)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 65)
        print("""
    Next steps:
    1. cd dashboard
    2. npm install  (first time only)
    3. npm run dev
    4. Open http://localhost:3000
        """)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure you're running from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
