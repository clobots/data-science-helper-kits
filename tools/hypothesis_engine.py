"""Hypothesis detection engine for identifying significant metric changes."""

from datetime import date, timedelta
from typing import Dict, List, Optional
import uuid

from config.settings import settings
from clients.hypothesis_client import HypothesisClient
from models.data_models import HierarchyLevel, MetricType, WeeklyComparison
from models.llm_inputs import HypothesisInput
from models.llm_outputs import HypothesisEngineOutput, HypothesisResult
from tools.data_extractor import DataExtractor
from utils.metrics import MetricsCalculator


class HypothesisEngine:
    """
    Engine for detecting significant changes in metrics.

    Analyzes data for:
    - Week-over-week changes
    - Budget variances
    - Trend anomalies
    """

    def __init__(
        self,
        data_extractor: Optional[DataExtractor] = None,
        use_llm: bool = True
    ):
        """
        Initialize the hypothesis engine.

        Args:
            data_extractor: DataExtractor instance (creates one if not provided)
            use_llm: Whether to use LLM for hypothesis generation
        """
        self.data_extractor = data_extractor or DataExtractor()
        self.metrics_calculator = MetricsCalculator()
        self.use_llm = use_llm

        if use_llm:
            self.client = HypothesisClient()
        else:
            self.client = None

    def detect_significant_changes(
        self,
        metrics: Optional[List[MetricType]] = None,
        hierarchy_level: HierarchyLevel = HierarchyLevel.STORE,
        current_week: Optional[date] = None,
        context: Optional[str] = None
    ) -> HypothesisEngineOutput:
        """
        Detect significant changes across metrics.

        Args:
            metrics: Metrics to analyze (defaults to all)
            hierarchy_level: Level to analyze at
            current_week: Current week ending date
            context: Additional business context

        Returns:
            HypothesisEngineOutput with detected hypotheses
        """
        metrics = metrics or [MetricType(m) for m in settings.metrics]

        # Gather all comparisons
        all_comparisons: List[WeeklyComparison] = []

        for metric in metrics:
            comparisons = self.data_extractor.get_weekly_comparisons(
                metric=metric,
                hierarchy_level=hierarchy_level,
                current_week=current_week
            )
            all_comparisons.extend(comparisons)

        # Determine analysis period
        if current_week is None and all_comparisons:
            current_week = all_comparisons[0].current_week

        start_date = current_week - timedelta(weeks=settings.num_weeks) if current_week else date.today() - timedelta(weeks=settings.num_weeks)
        end_date = current_week or date.today()

        if self.use_llm and self.client:
            # Use LLM for hypothesis generation
            input_data = HypothesisInput(
                metric_comparisons=all_comparisons,
                analysis_period={"start": start_date, "end": end_date},
                significance_thresholds={
                    "wow_change_pct": settings.thresholds.wow_change_pct,
                    "budget_variance_pct": settings.thresholds.budget_variance_pct,
                    "trend_deviation_pct": settings.thresholds.trend_deviation_pct
                },
                context=context
            )

            return self.client.detect_hypotheses(input_data, current_week)
        else:
            # Use rule-based detection
            return self._detect_rule_based(all_comparisons, current_week or date.today())

    def _detect_rule_based(
        self,
        comparisons: List[WeeklyComparison],
        analysis_date: date
    ) -> HypothesisEngineOutput:
        """
        Rule-based hypothesis detection without LLM.

        Args:
            comparisons: Weekly comparison data
            analysis_date: Date of analysis

        Returns:
            HypothesisEngineOutput with detected hypotheses
        """
        hypotheses: List[HypothesisResult] = []

        for comp in comparisons:
            # Check for significant WoW changes
            if self.metrics_calculator.is_significant_wow_change(
                comp.wow_change_pct,
                comp.metric_name
            ):
                change_type = "wow_increase" if comp.wow_change_pct > 0 else "wow_decrease"
                hypothesis = self._create_hypothesis(
                    comp, change_type, comp.wow_change_pct
                )
                hypotheses.append(hypothesis)

            # Check for significant budget variances
            if comp.budget_variance_pct is not None:
                if self.metrics_calculator.is_significant_budget_variance(
                    comp.budget_variance_pct,
                    comp.metric_name
                ):
                    change_type = "budget_over" if comp.budget_variance_pct > 0 else "budget_under"
                    hypothesis = self._create_hypothesis(
                        comp, change_type, comp.budget_variance_pct
                    )
                    hypotheses.append(hypothesis)

        # Sort by significance and limit
        hypotheses.sort(key=lambda h: h.significance_score, reverse=True)
        hypotheses = hypotheses[:20]  # Limit to top 20

        # Generate summary
        if hypotheses:
            high_priority = len([h for h in hypotheses if h.significance_score > 0.7])
            summary = (
                f"Detected {len(hypotheses)} significant changes. "
                f"{high_priority} are high priority requiring immediate investigation."
            )
        else:
            summary = "No significant changes detected in the analyzed period."

        return HypothesisEngineOutput(
            analysis_date=analysis_date,
            total_comparisons_analyzed=len(comparisons),
            hypotheses=hypotheses,
            summary=summary
        )

    def _create_hypothesis(
        self,
        comparison: WeeklyComparison,
        change_type: str,
        change_value: float
    ) -> HypothesisResult:
        """Create a hypothesis from a comparison."""
        # Calculate significance score
        significance = self.metrics_calculator.calculate_significance_score(
            comparison.wow_change_pct,
            comparison.budget_variance_pct,
            None  # No trend deviation for simple detection
        )

        # Generate description
        if self.client:
            description = self.client.generate_hypothesis_description(
                metric=comparison.metric_name,
                change_type=change_type,
                change_value=change_value,
                hierarchy_level=comparison.hierarchy_level,
                hierarchy_name=comparison.hierarchy_name
            )
        else:
            description = self._generate_description(
                comparison, change_type, change_value
            )

        return HypothesisResult(
            id=f"HYP-{uuid.uuid4().hex[:8].upper()}",
            metric=comparison.metric_name,
            hierarchy_level=comparison.hierarchy_level,
            hierarchy_id=comparison.hierarchy_id,
            hierarchy_name=comparison.hierarchy_name,
            change_type=change_type,
            change_value=change_value,
            change_percentage=change_value,
            significance_score=significance,
            description=description,
            requires_investigation=significance > 0.5
        )

    def _generate_description(
        self,
        comparison: WeeklyComparison,
        change_type: str,
        change_value: float
    ) -> str:
        """Generate a simple hypothesis description."""
        metric_name = comparison.metric_name.replace("_", " ").title()

        change_descriptions = {
            "wow_increase": f"increased by {abs(change_value):.1f}%",
            "wow_decrease": f"decreased by {abs(change_value):.1f}%",
            "budget_over": f"is {abs(change_value):.1f}% over budget",
            "budget_under": f"is {abs(change_value):.1f}% under budget",
        }

        change_text = change_descriptions.get(
            change_type,
            f"changed by {change_value:+.1f}%"
        )

        return (
            f"{metric_name} at {comparison.hierarchy_name} ({comparison.hierarchy_level}) "
            f"has {change_text} week-over-week"
        )

    def validate_hypothesis(
        self,
        hypothesis: HypothesisResult
    ) -> Dict:
        """
        Validate a hypothesis with additional data.

        Args:
            hypothesis: The hypothesis to validate

        Returns:
            Validation result with confidence and recommendations
        """
        if not self.use_llm or not self.client:
            return {
                "is_valid": hypothesis.significance_score > 0.5,
                "confidence": hypothesis.significance_score,
                "concerns": [],
                "recommended_action": "investigate" if hypothesis.significance_score > 0.5 else "monitor"
            }

        # Get additional data for validation
        comparisons = self.data_extractor.get_weekly_comparisons(
            metric=hypothesis.metric,
            hierarchy_level=hypothesis.hierarchy_level
        )

        supporting_data = {
            "total_entities_at_level": len(comparisons),
            "entities_with_similar_change": len([
                c for c in comparisons
                if abs(c.wow_change_pct - hypothesis.change_percentage) < 2.0
            ]),
            "average_change_at_level": sum(c.wow_change_pct for c in comparisons) / len(comparisons) if comparisons else 0
        }

        return self.client.validate_hypothesis(hypothesis, supporting_data)

    def prioritize_hypotheses(
        self,
        hypotheses: List[HypothesisResult]
    ) -> List[HypothesisResult]:
        """
        Prioritize hypotheses for investigation.

        Args:
            hypotheses: List of hypotheses to prioritize

        Returns:
            Sorted list of hypotheses by priority
        """
        if self.use_llm and self.client:
            prioritized = self.client.prioritize_hypotheses(hypotheses)

            # Create a priority map
            priority_map = {
                p["id"]: p["priority_rank"]
                for p in prioritized
            }

            # Sort hypotheses by priority
            return sorted(
                hypotheses,
                key=lambda h: priority_map.get(h.id, 999)
            )
        else:
            # Simple sort by significance
            return sorted(
                hypotheses,
                key=lambda h: h.significance_score,
                reverse=True
            )

    def get_related_hypotheses(
        self,
        hypothesis: HypothesisResult,
        all_hypotheses: List[HypothesisResult]
    ) -> List[HypothesisResult]:
        """
        Find hypotheses related to a given one.

        Args:
            hypothesis: The reference hypothesis
            all_hypotheses: All hypotheses to search

        Returns:
            Related hypotheses
        """
        related = []

        for h in all_hypotheses:
            if h.id == hypothesis.id:
                continue

            # Same metric at different level
            if h.metric == hypothesis.metric:
                related.append(h)
            # Same location, different metric
            elif h.hierarchy_id == hypothesis.hierarchy_id:
                related.append(h)

        return related
