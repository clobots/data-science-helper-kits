"""Insights generation engine for hierarchical investigation of hypotheses."""

from datetime import date, timedelta
from typing import Dict, List, Optional

from config.settings import settings
from clients.insights_client import InsightsClient
from models.data_models import HierarchyLevel, MetricType
from models.llm_inputs import InsightsInput
from models.llm_outputs import HypothesisResult, InsightResult, InsightsEngineOutput
from tools.data_extractor import DataExtractor
from utils.metrics import MetricsCalculator


class InsightsEngine:
    """
    Engine for generating hierarchical insights from hypotheses.

    Investigates hypotheses by drilling through the hierarchy:
    Category → Store → Group → State → Zone → National
    """

    # Hierarchy drill paths (from specific to broad)
    DRILL_UP_ORDER = [
        HierarchyLevel.CATEGORY,
        HierarchyLevel.STORE,
        HierarchyLevel.GROUP,
        HierarchyLevel.STATE,
        HierarchyLevel.ZONE,
        HierarchyLevel.NATIONAL
    ]

    def __init__(
        self,
        data_extractor: Optional[DataExtractor] = None,
        use_llm: bool = True
    ):
        """
        Initialize the insights engine.

        Args:
            data_extractor: DataExtractor instance
            use_llm: Whether to use LLM for insight generation
        """
        self.data_extractor = data_extractor or DataExtractor()
        self.metrics_calculator = MetricsCalculator()
        self.use_llm = use_llm

        if use_llm:
            self.client = InsightsClient()
        else:
            self.client = None

    def investigate_hypothesis(
        self,
        hypothesis: HypothesisResult,
        max_levels: int = 4,
        current_week: Optional[date] = None
    ) -> InsightsEngineOutput:
        """
        Investigate a hypothesis through hierarchy levels.

        Args:
            hypothesis: The hypothesis to investigate
            max_levels: Maximum levels to drill through
            current_week: Current week ending date

        Returns:
            InsightsEngineOutput with complete investigation
        """
        insights: List[InsightResult] = []
        investigation_path: List[str] = []

        # Start from the hypothesis level
        start_level_idx = self.DRILL_UP_ORDER.index(hypothesis.hierarchy_level)

        # Drill both up and down from the starting level
        levels_to_investigate = []

        # Add levels going up (to broader levels)
        for i in range(start_level_idx, min(start_level_idx + max_levels, len(self.DRILL_UP_ORDER))):
            levels_to_investigate.append(self.DRILL_UP_ORDER[i])

        # Also investigate one level down if possible
        if start_level_idx > 0:
            levels_to_investigate.insert(0, self.DRILL_UP_ORDER[start_level_idx - 1])

        parent_insight: Optional[str] = None

        for level in levels_to_investigate:
            # Get data at this level
            hierarchy_data = self.data_extractor.get_hierarchy_data(
                metric=hypothesis.metric,
                level=level,
                current_week=current_week or date.today()
            )

            if not hierarchy_data:
                continue

            # Generate insight for this level
            insight = self._generate_insight_for_level(
                hypothesis=hypothesis,
                level=level,
                hierarchy_data=hierarchy_data,
                parent_insight=parent_insight,
                investigation_path=investigation_path.copy()
            )

            insights.append(insight)
            investigation_path.append(level.value)
            parent_insight = insight.finding

            # Stop if we found a likely root cause
            if insight.root_cause_likelihood > 0.8:
                break

        # Build complete output
        if self.use_llm and self.client:
            return self.client.build_complete_investigation(
                hypothesis_id=hypothesis.id,
                hypothesis_description=hypothesis.description,
                insights=insights
            )
        else:
            return self._build_investigation_output(
                hypothesis=hypothesis,
                insights=insights,
                investigation_path=investigation_path
            )

    def _generate_insight_for_level(
        self,
        hypothesis: HypothesisResult,
        level: HierarchyLevel,
        hierarchy_data: List[Dict],
        parent_insight: Optional[str],
        investigation_path: List[str]
    ) -> InsightResult:
        """Generate an insight for a specific hierarchy level."""
        if self.use_llm and self.client:
            input_data = InsightsInput(
                hypothesis_description=hypothesis.description,
                metric_name=hypothesis.metric,
                hierarchy_level=level,
                hierarchy_data=hierarchy_data,
                parent_insight=parent_insight,
                drill_down_path=investigation_path
            )

            return self.client.drill_down(input_data, hypothesis.id)
        else:
            return self._generate_rule_based_insight(
                hypothesis=hypothesis,
                level=level,
                hierarchy_data=hierarchy_data
            )

    def _generate_rule_based_insight(
        self,
        hypothesis: HypothesisResult,
        level: HierarchyLevel,
        hierarchy_data: List[Dict]
    ) -> InsightResult:
        """Generate insight using rule-based analysis."""
        # Find the entity with the largest contribution to the issue
        sorted_data = sorted(
            hierarchy_data,
            key=lambda x: abs(x.get("wow_change_pct", 0)),
            reverse=True
        )

        top_entity = sorted_data[0] if sorted_data else {}

        # Calculate contribution percentage
        total_change = sum(abs(d.get("wow_change_pct", 0)) for d in hierarchy_data)
        top_contribution = abs(top_entity.get("wow_change_pct", 0))
        contribution_pct = (top_contribution / total_change * 100) if total_change > 0 else 0

        # Determine if this looks like a root cause
        # High concentration = likely root cause
        root_cause_likelihood = min(contribution_pct / 100 * 1.5, 1.0)

        # Count how many entities are affected
        threshold = settings.thresholds.wow_change_pct
        affected_count = len([
            d for d in hierarchy_data
            if abs(d.get("wow_change_pct", 0)) > threshold
        ])

        # Generate finding
        if contribution_pct > 50:
            finding = (
                f"Issue is concentrated in {top_entity.get('name', 'Unknown')} "
                f"which accounts for {contribution_pct:.1f}% of the total change"
            )
        elif affected_count > len(hierarchy_data) * 0.5:
            finding = (
                f"Issue is widespread at {level.value} level with "
                f"{affected_count} of {len(hierarchy_data)} entities affected"
            )
        else:
            finding = (
                f"Mixed pattern at {level.value} level - "
                f"{affected_count} entities show significant changes"
            )

        return InsightResult(
            hypothesis_id=hypothesis.id,
            hierarchy_level=level,
            hierarchy_id=top_entity.get("id", "unknown"),
            hierarchy_name=top_entity.get("name", "Unknown"),
            finding=finding,
            contribution_percentage=contribution_pct,
            root_cause_likelihood=root_cause_likelihood,
            supporting_data={
                "top_contributors": sorted_data[:5],
                "total_entities": len(hierarchy_data),
                "affected_count": affected_count
            },
            recommended_actions=self._generate_recommendations(
                level, contribution_pct, affected_count
            ),
            requires_deeper_investigation=root_cause_likelihood < 0.6
        )

    def _generate_recommendations(
        self,
        level: HierarchyLevel,
        contribution_pct: float,
        affected_count: int
    ) -> List[str]:
        """Generate recommendations based on insight patterns."""
        recommendations = []

        if contribution_pct > 50:
            # Concentrated issue
            recommendations.append(
                f"Focus investigation on the top contributing {level.value}"
            )
            recommendations.append(
                "Review recent changes specific to this entity"
            )
        elif affected_count > 3:
            # Widespread issue
            recommendations.append(
                "Investigate systemic factors affecting multiple entities"
            )
            recommendations.append(
                "Check for common operational or environmental changes"
            )
        else:
            # Mixed pattern
            recommendations.append(
                "Compare affected vs unaffected entities for differences"
            )

        return recommendations

    def _build_investigation_output(
        self,
        hypothesis: HypothesisResult,
        insights: List[InsightResult],
        investigation_path: List[str]
    ) -> InsightsEngineOutput:
        """Build the complete investigation output without LLM."""
        # Find the insight with highest root cause likelihood
        if insights:
            best_insight = max(insights, key=lambda i: i.root_cause_likelihood)
            root_cause_summary = best_insight.finding
            confidence = best_insight.root_cause_likelihood
        else:
            root_cause_summary = "Unable to determine root cause"
            confidence = 0.0

        # Aggregate recommendations
        all_recommendations = []
        for insight in insights:
            all_recommendations.extend(insight.recommended_actions)
        # Deduplicate
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        return InsightsEngineOutput(
            hypothesis_id=hypothesis.id,
            hypothesis_description=hypothesis.description,
            investigation_path=investigation_path,
            insights=insights,
            root_cause_summary=root_cause_summary,
            confidence_score=confidence,
            overall_recommendations=unique_recommendations[:5]
        )

    def investigate_multiple(
        self,
        hypotheses: List[HypothesisResult],
        max_levels: int = 3,
        current_week: Optional[date] = None
    ) -> List[InsightsEngineOutput]:
        """
        Investigate multiple hypotheses.

        Args:
            hypotheses: List of hypotheses to investigate
            max_levels: Maximum levels per hypothesis
            current_week: Current week ending date

        Returns:
            List of investigation outputs
        """
        results = []
        for hypothesis in hypotheses:
            result = self.investigate_hypothesis(
                hypothesis=hypothesis,
                max_levels=max_levels,
                current_week=current_week
            )
            results.append(result)
        return results

    def analyze_cross_hypothesis_patterns(
        self,
        investigations: List[InsightsEngineOutput]
    ) -> Dict:
        """
        Analyze patterns across multiple hypothesis investigations.

        Args:
            investigations: Completed investigations

        Returns:
            Cross-hypothesis analysis
        """
        if not investigations:
            return {"patterns": [], "common_factors": []}

        # Collect all insights
        all_insights = []
        for inv in investigations:
            all_insights.extend(inv.insights)

        # Find common affected entities
        entity_counts: Dict[str, int] = {}
        for insight in all_insights:
            key = f"{insight.hierarchy_level}:{insight.hierarchy_id}"
            entity_counts[key] = entity_counts.get(key, 0) + 1

        # Entities appearing in multiple investigations
        common_entities = [
            entity for entity, count in entity_counts.items()
            if count > 1
        ]

        # Find common recommendations
        all_recommendations = []
        for inv in investigations:
            all_recommendations.extend(inv.overall_recommendations)

        recommendation_counts: Dict[str, int] = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        common_recommendations = [
            rec for rec, count in recommendation_counts.items()
            if count > 1
        ]

        return {
            "patterns": [
                {
                    "type": "common_entity",
                    "entities": common_entities,
                    "description": f"{len(common_entities)} entities appear in multiple investigations"
                }
            ],
            "common_factors": common_entities,
            "unified_recommendations": common_recommendations
        }
