#!/usr/bin/env python3
"""
Data Science Agent for Woolworths Australia Analytics

This is the main entry point for the data science agent.
Run this script to execute a demo analysis workflow.
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.data_science_agent import DataScienceAgent
from config.settings import settings
from data.generator import WoolworthsDataGenerator
from models.data_models import HierarchyLevel, MetricType


def generate_sample_data(force: bool = False) -> None:
    """Generate sample data if it doesn't exist."""
    data_path = settings.data_path

    if force or not data_path.exists() or not any(data_path.glob("*.csv")):
        print("Generating sample data...")
        generator = WoolworthsDataGenerator(seed=42)
        saved_files = generator.save_data(data_path)
        print(f"Generated {len(saved_files)} data files in {data_path}")
    else:
        print(f"Sample data already exists in {data_path}")


def run_demo(use_llm: bool = False) -> None:
    """Run a demonstration of the data science agent."""
    print("\n" + "=" * 60)
    print("WOOLWORTHS DATA SCIENCE AGENT - DEMO")
    print("=" * 60)

    # Ensure data exists
    generate_sample_data()

    # Initialize agent
    print("\nInitializing agent...")
    agent = DataScienceAgent(use_llm=use_llm)

    # Run quick scan first
    print("\n" + "-" * 40)
    print("QUICK SCAN")
    print("-" * 40)
    scan_result = agent.quick_scan()
    print(f"Total issues detected: {scan_result['total_detected']}")
    print(f"Critical issues: {scan_result['critical_count']}")
    print(f"Recommendation: {scan_result['recommendation']}")

    # Run full analysis
    print("\n" + "-" * 40)
    print("FULL ANALYSIS")
    print("-" * 40)
    result = agent.run_full_analysis(
        metrics=[MetricType.SALES, MetricType.WAGES, MetricType.VOICE_OF_CUSTOMER],
        hierarchy_level=HierarchyLevel.STORE,
        audience="executive",
        tone="formal",
        max_hypotheses=3,
        max_investigation_levels=2
    )

    # Display results
    print("\n" + "-" * 40)
    print("ANALYSIS RESULTS")
    print("-" * 40)
    print(f"\nAnalysis Date: {result.analysis_date}")
    print(f"Hypotheses Found: {len(result.hypotheses_output.hypotheses)}")
    print(f"Investigations Completed: {len(result.investigations)}")
    print(f"Critical Findings: {result.has_critical_findings}")

    print("\n--- EXECUTIVE SUMMARY ---")
    print(result.summary.executive_summary)

    print("\n--- KEY FINDINGS ---")
    for finding in result.summary.key_findings[:3]:
        print(f"• [{finding.impact.upper()}] {finding.title}")
        print(f"  {finding.description}")

    print("\n--- TOP RECOMMENDATIONS ---")
    for i, rec in enumerate(result.top_recommendations[:3], 1):
        print(f"{i}. {rec}")

    if result.alert:
        print("\n--- ALERT ---")
        print(f"Severity: {result.alert.get('severity', 'N/A')}")
        print(f"Headline: {result.alert.get('headline', 'N/A')}")

    # Generate markdown report
    print("\n" + "-" * 40)
    print("GENERATING REPORT")
    print("-" * 40)
    report = agent.generate_report(result, format_type="markdown")
    report_path = settings.base_path / "output" / "analysis_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")

    # Demo standalone summariser
    print("\n" + "-" * 40)
    print("STANDALONE SUMMARISER DEMO")
    print("-" * 40)
    sample_content = """
    This week's sales performance showed a significant uptick in the Fresh Produce
    category, with a 12% increase compared to last week. However, the Frozen Foods
    category experienced a 5% decline, potentially due to supply chain disruptions.
    Customer satisfaction scores remained stable at 82/100, though there were
    isolated complaints about checkout wait times in NSW Metro stores.
    Wages as a percentage of sales improved by 0.3 percentage points, indicating
    better labor efficiency. Stock loss remained within acceptable limits at 1.1%.
    """

    summary = agent.summarise_content(
        content=sample_content,
        audience_goal="Understand weekly performance highlights",
        expertise="medium",
        tone="conversational"
    )

    print("Summary:")
    print(summary.summary)
    print("\nKey Points:")
    for point in summary.key_points:
        print(f"• {point}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def run_analysis(
    metrics: list,
    audience: str,
    tone: str,
    use_llm: bool
) -> None:
    """Run a custom analysis."""
    generate_sample_data()

    agent = DataScienceAgent(use_llm=use_llm)

    metric_types = [MetricType(m) for m in metrics]

    result = agent.run_full_analysis(
        metrics=metric_types,
        audience=audience,
        tone=tone
    )

    report = agent.generate_report(result, format_type="markdown")
    print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Woolworths Data Science Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    Run demo workflow
  python main.py --demo --use-llm          Run demo with Gemini LLM
  python main.py --generate-data           Generate sample data only
  python main.py --metrics sales wages     Analyze specific metrics
  python main.py --quick-scan              Run quick scan only
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the demo workflow"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Gemini LLM for analysis (requires GEMINI_API_KEY)"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate sample data only"
    )

    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regenerate sample data"
    )

    parser.add_argument(
        "--quick-scan",
        action="store_true",
        help="Run quick scan only"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["sales", "wages", "voice_of_customer", "stockloss", "order_pickrate"],
        default=["sales", "wages", "voice_of_customer"],
        help="Metrics to analyze"
    )

    parser.add_argument(
        "--audience",
        choices=["executive", "analyst", "operations"],
        default="executive",
        help="Target audience for summary"
    )

    parser.add_argument(
        "--tone",
        choices=["formal", "conversational", "urgent"],
        default="formal",
        help="Tone for summary"
    )

    args = parser.parse_args()

    if args.generate_data:
        generate_sample_data(force=args.force_regenerate)
        return

    if args.quick_scan:
        generate_sample_data()
        agent = DataScienceAgent(use_llm=args.use_llm)
        result = agent.quick_scan()
        import json
        print(json.dumps(result, indent=2))
        return

    if args.demo:
        run_demo(use_llm=args.use_llm)
        return

    # Default: run analysis
    run_analysis(
        metrics=args.metrics,
        audience=args.audience,
        tone=args.tone,
        use_llm=args.use_llm
    )


if __name__ == "__main__":
    main()
