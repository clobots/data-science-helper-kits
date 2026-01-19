"""Main data science agent orchestrating all tools."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Literal, Optional

from config.settings import settings
from models.data_models import HierarchyLevel, MetricType
from models.llm_outputs import (
    HypothesisEngineOutput,
    HypothesisResult,
    InsightsEngineOutput,
    SummaryOutput,
    SummariserOutput,
)
from tools.data_extractor import DataExtractor
from tools.hypothesis_engine import HypothesisEngine
from tools.insights_engine import InsightsEngine
from tools.summariser import Summariser


@dataclass
class AnalysisResult:
    """Complete result from an analysis run."""

    analysis_date: date
    hypotheses_output: HypothesisEngineOutput
    investigations: List[InsightsEngineOutput]
    summary: SummaryOutput
    alert: Optional[Dict] = None
    metrics: Dict = field(default_factory=dict)

    @property
    def has_critical_findings(self) -> bool:
        """Check if there are any critical findings."""
        return any(
            h.significance_score > 0.8
            for h in self.hypotheses_output.hypotheses
        )

    @property
    def top_recommendations(self) -> List[str]:
        """Get top recommendations across all investigations."""
        all_recs = []
        for inv in self.investigations:
            all_recs.extend(inv.overall_recommendations)
        # Deduplicate while preserving order
        seen = set()
        return [r for r in all_recs if not (r in seen or seen.add(r))][:5]


class DataScienceAgent:
    """
    Main agent for Woolworths data science analysis.

    Orchestrates:
    1. Data extraction from simulated BigQuery
    2. Hypothesis detection for significant changes
    3. Hierarchical insights generation
    4. Audience-specific summary creation
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the data science agent.

        Args:
            use_llm: Whether to use LLM for analysis (False for demo/testing)
        """
        self.use_llm = use_llm

        # Initialize tools
        self.data_extractor = DataExtractor()
        self.hypothesis_engine = HypothesisEngine(
            data_extractor=self.data_extractor,
            use_llm=use_llm
        )
        self.insights_engine = InsightsEngine(
            data_extractor=self.data_extractor,
            use_llm=use_llm
        )
        self.summariser = Summariser(use_llm=use_llm)

    def run_full_analysis(
        self,
        metrics: Optional[List[MetricType]] = None,
        hierarchy_level: HierarchyLevel = HierarchyLevel.STORE,
        audience: Literal["executive", "analyst", "operations"] = "executive",
        tone: Literal["formal", "conversational", "urgent"] = "formal",
        max_hypotheses: int = 5,
        max_investigation_levels: int = 3,
        current_week: Optional[date] = None,
        context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run a complete analysis pipeline.

        Args:
            metrics: Metrics to analyze (defaults to all)
            hierarchy_level: Starting hierarchy level
            audience: Target audience for summary
            tone: Tone for summary
            max_hypotheses: Maximum hypotheses to investigate
            max_investigation_levels: Maximum levels to drill per hypothesis
            current_week: Current week ending date
            context: Additional business context

        Returns:
            AnalysisResult with all findings
        """
        print("Starting full analysis pipeline...")

        # Step 1: Detect hypotheses
        print("\n[1/4] Detecting significant changes...")
        hypotheses_output = self.hypothesis_engine.detect_significant_changes(
            metrics=metrics,
            hierarchy_level=hierarchy_level,
            current_week=current_week,
            context=context
        )
        print(f"  Found {len(hypotheses_output.hypotheses)} hypotheses")

        # Step 2: Prioritize and select top hypotheses
        print("\n[2/4] Prioritizing hypotheses...")
        prioritized = self.hypothesis_engine.prioritize_hypotheses(
            hypotheses_output.hypotheses
        )
        top_hypotheses = prioritized[:max_hypotheses]
        print(f"  Selected top {len(top_hypotheses)} for investigation")

        # Step 3: Investigate each hypothesis
        print("\n[3/4] Investigating hypotheses...")
        investigations = []
        for i, hypothesis in enumerate(top_hypotheses, 1):
            print(f"  Investigating {i}/{len(top_hypotheses)}: {hypothesis.metric}")
            investigation = self.insights_engine.investigate_hypothesis(
                hypothesis=hypothesis,
                max_levels=max_investigation_levels,
                current_week=current_week
            )
            investigations.append(investigation)

        # Step 4: Generate summary
        print("\n[4/4] Generating summary...")
        summary = self.summariser.generate_analysis_summary(
            hypotheses=top_hypotheses,
            investigations=investigations,
            audience=audience,
            tone=tone
        )

        # Check for alerts
        alert = self.summariser.generate_alert(
            hypotheses=top_hypotheses,
            investigations=investigations
        )

        # Extract metrics for dashboard
        metrics_summary = self.summariser.extract_key_metrics(investigations)

        result = AnalysisResult(
            analysis_date=current_week or date.today(),
            hypotheses_output=hypotheses_output,
            investigations=investigations,
            summary=summary,
            alert=alert,
            metrics=metrics_summary
        )

        print("\nAnalysis complete!")
        if result.has_critical_findings:
            print("WARNING: Critical findings detected!")

        return result

    def analyze_single_metric(
        self,
        metric: MetricType,
        hierarchy_level: HierarchyLevel = HierarchyLevel.STORE,
        current_week: Optional[date] = None
    ) -> AnalysisResult:
        """
        Analyze a single metric in detail.

        Args:
            metric: The metric to analyze
            hierarchy_level: Starting hierarchy level
            current_week: Current week ending date

        Returns:
            AnalysisResult focused on one metric
        """
        return self.run_full_analysis(
            metrics=[metric],
            hierarchy_level=hierarchy_level,
            max_hypotheses=10,  # More hypotheses for single metric
            max_investigation_levels=4,  # Deeper investigation
            current_week=current_week,
            context=f"Focused analysis on {metric.value}"
        )

    def quick_scan(
        self,
        current_week: Optional[date] = None
    ) -> Dict:
        """
        Quick scan for critical issues only.

        Args:
            current_week: Current week ending date

        Returns:
            Dictionary with critical findings
        """
        print("Running quick scan...")

        hypotheses_output = self.hypothesis_engine.detect_significant_changes(
            hierarchy_level=HierarchyLevel.NATIONAL,
            current_week=current_week
        )

        critical = [
            h for h in hypotheses_output.hypotheses
            if h.significance_score > 0.7
        ]

        return {
            "scan_date": date.today().isoformat(),
            "total_detected": len(hypotheses_output.hypotheses),
            "critical_count": len(critical),
            "critical_issues": [
                {
                    "metric": h.metric,
                    "description": h.description,
                    "significance": h.significance_score
                }
                for h in critical
            ],
            "recommendation": "Run full analysis" if critical else "No immediate action needed"
        }

    def summarise_content(
        self,
        content: str,
        audience_goal: str,
        expertise: Literal["low", "medium", "high"] = "medium",
        tone: Literal["formal", "conversational", "urgent", "encouraging"] = "formal"
    ) -> SummariserOutput:
        """
        Summarise arbitrary content (standalone summariser tool).

        Args:
            content: Content to summarise
            audience_goal: What audience needs from summary
            expertise: Audience expertise level
            tone: Desired tone

        Returns:
            SummariserOutput with summary
        """
        return self.summariser.summarise(
            content=content,
            audience_goal=audience_goal,
            expertise=expertise,
            tone=tone
        )

    def get_data_snapshot(
        self,
        metric: MetricType,
        hierarchy_level: HierarchyLevel,
        weeks: int = 4
    ) -> Dict:
        """
        Get a snapshot of recent data for a metric.

        Args:
            metric: Metric to retrieve
            hierarchy_level: Level of aggregation
            weeks: Number of weeks of history

        Returns:
            Data snapshot dictionary
        """
        end_date = date.today()
        start_date = end_date - timedelta(weeks=weeks)

        df = self.data_extractor.get_aggregated_data(
            metric=metric,
            hierarchy_level=hierarchy_level,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "metric": metric.value,
            "hierarchy_level": hierarchy_level.value,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary_stats": {
                "mean": df["value"].mean() if not df.empty else 0,
                "std": df["value"].std() if not df.empty else 0,
                "min": df["value"].min() if not df.empty else 0,
                "max": df["value"].max() if not df.empty else 0
            },
            "data_points": len(df)
        }

    def compare_periods(
        self,
        metric: MetricType,
        hierarchy_level: HierarchyLevel,
        current_weeks: int = 4,
        comparison_weeks: int = 4
    ) -> Dict:
        """
        Compare two time periods for a metric.

        Args:
            metric: Metric to compare
            hierarchy_level: Level of aggregation
            current_weeks: Weeks in current period
            comparison_weeks: Weeks in comparison period

        Returns:
            Period comparison results
        """
        end_date = date.today()
        current_start = end_date - timedelta(weeks=current_weeks)
        comparison_end = current_start - timedelta(days=1)
        comparison_start = comparison_end - timedelta(weeks=comparison_weeks)

        current_df = self.data_extractor.get_aggregated_data(
            metric=metric,
            hierarchy_level=hierarchy_level,
            start_date=current_start,
            end_date=end_date
        )

        comparison_df = self.data_extractor.get_aggregated_data(
            metric=metric,
            hierarchy_level=hierarchy_level,
            start_date=comparison_start,
            end_date=comparison_end
        )

        current_mean = current_df["value"].mean() if not current_df.empty else 0
        comparison_mean = comparison_df["value"].mean() if not comparison_df.empty else 0

        change = current_mean - comparison_mean
        change_pct = (change / comparison_mean * 100) if comparison_mean != 0 else 0

        return {
            "metric": metric.value,
            "current_period": {
                "start": current_start.isoformat(),
                "end": end_date.isoformat(),
                "mean": current_mean
            },
            "comparison_period": {
                "start": comparison_start.isoformat(),
                "end": comparison_end.isoformat(),
                "mean": comparison_mean
            },
            "change": {
                "absolute": change,
                "percentage": change_pct,
                "direction": "up" if change > 0 else "down" if change < 0 else "flat"
            }
        }

    def generate_report(
        self,
        result: AnalysisResult,
        format_type: Literal["text", "json", "markdown"] = "markdown"
    ) -> str:
        """
        Generate a formatted report from analysis results.

        Args:
            result: AnalysisResult from analysis run
            format_type: Output format

        Returns:
            Formatted report string
        """
        if format_type == "json":
            import json
            return json.dumps({
                "analysis_date": result.analysis_date.isoformat(),
                "summary": result.summary.executive_summary,
                "hypotheses_count": len(result.hypotheses_output.hypotheses),
                "investigations_count": len(result.investigations),
                "has_critical": result.has_critical_findings,
                "recommendations": result.top_recommendations,
                "alert": result.alert
            }, indent=2)

        elif format_type == "markdown":
            lines = [
                f"# Woolworths Analytics Report",
                f"**Date:** {result.analysis_date.isoformat()}",
                "",
                "## Executive Summary",
                result.summary.executive_summary,
                "",
                "## Key Findings",
            ]

            for finding in result.summary.key_findings:
                lines.append(f"### {finding.title}")
                lines.append(f"- **Impact:** {finding.impact}")
                lines.append(f"- {finding.description}")
                lines.append("")

            lines.append("## Recommended Actions")
            for i, action in enumerate(result.summary.action_items, 1):
                lines.append(f"{i}. **[{action.priority}]** {action.action}")

            if result.alert:
                lines.extend([
                    "",
                    "## Alert",
                    f"**Severity:** {result.alert.get('severity', 'unknown')}",
                    f"**{result.alert.get('headline', 'Alert generated')}"
                ])

            return "\n".join(lines)

        else:  # text
            lines = [
                "=" * 50,
                "WOOLWORTHS ANALYTICS REPORT",
                f"Date: {result.analysis_date.isoformat()}",
                "=" * 50,
                "",
                "EXECUTIVE SUMMARY",
                "-" * 30,
                result.summary.executive_summary,
                "",
                "RECOMMENDATIONS",
                "-" * 30,
            ]

            for i, rec in enumerate(result.top_recommendations, 1):
                lines.append(f"  {i}. {rec}")

            return "\n".join(lines)
