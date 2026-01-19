"""Client for insights generation LLM interactions."""

from typing import Dict, List, Optional

from config.settings import settings
from models.data_models import HierarchyLevel
from models.llm_inputs import InsightsInput
from models.llm_outputs import InsightResult, InsightsEngineOutput

from .base_client import BaseGeminiClient


class InsightsClient(BaseGeminiClient):
    """Client for generating hierarchical insights using Gemini."""

    def __init__(self, temperature: Optional[float] = None):
        """Initialize with analytical temperature setting."""
        prompts = settings.load_prompt("insights.yaml")
        system_prompt = prompts.get("system_prompt", "")

        super().__init__(
            temperature=temperature or settings.temperature_analytical,
            system_prompt=system_prompt
        )
        self.prompts = prompts

    def drill_down(
        self,
        input_data: InsightsInput,
        hypothesis_id: str
    ) -> InsightResult:
        """
        Analyze data at a specific hierarchy level.

        Args:
            input_data: InsightsInput with hypothesis and level data
            hypothesis_id: ID of the hypothesis being investigated

        Returns:
            InsightResult with findings at this level
        """
        context = input_data.to_prompt_context()
        prompt_template = self.prompts.get("drill_down_prompt", "")

        prompt = prompt_template.format(
            hypothesis=input_data.hypothesis_description,
            hypothesis_id=hypothesis_id,
            current_level=input_data.hierarchy_level,
            path=" -> ".join(input_data.drill_down_path) if input_data.drill_down_path else "Starting",
            parent_insight=input_data.parent_insight or "N/A",
            data=str(input_data.hierarchy_data[:15])  # Limit data for context
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            InsightResult,
            strict=False
        )

    def analyze_patterns(
        self,
        insights: List[InsightResult]
    ) -> Dict:
        """
        Analyze patterns across multiple hierarchy levels.

        Args:
            insights: List of insights from different levels

        Returns:
            Pattern analysis with root cause hypothesis
        """
        insights_text = "\n".join([
            f"- {i.hierarchy_level}: {i.finding} (contribution: {i.contribution_percentage:.1f}%)"
            for i in insights
        ])

        prompt_template = self.prompts.get("pattern_analysis_prompt", "")
        prompt = prompt_template.format(insights=insights_text)

        response = self.invoke(prompt)
        return self.parser.parse_json(response)

    def generate_recommendations(
        self,
        hypothesis_description: str,
        summary: str,
        insights: List[InsightResult]
    ) -> List[Dict]:
        """
        Generate actionable recommendations from insights.

        Args:
            hypothesis_description: The investigated hypothesis
            summary: Summary of investigation
            insights: All insights gathered

        Returns:
            List of recommendations with priorities
        """
        insights_text = "\n".join([
            f"- {i.hierarchy_level} ({i.hierarchy_name}): {i.finding}"
            for i in insights
        ])

        prompt_template = self.prompts.get("recommendation_prompt", "")
        prompt = prompt_template.format(
            hypothesis=hypothesis_description,
            summary=summary,
            insights=insights_text
        )

        response = self.invoke(prompt)
        result = self.parser.parse_json(response)

        return result.get("recommendations", [])

    def analyze_cross_metric(
        self,
        metrics_data: Dict[str, List]
    ) -> Dict:
        """
        Analyze correlations between different metrics.

        Args:
            metrics_data: Dictionary of metric name to values

        Returns:
            Correlation analysis results
        """
        data_text = "\n".join([
            f"- {metric}: {values[:10]}..."
            for metric, values in metrics_data.items()
        ])

        prompt_template = self.prompts.get("cross_metric_prompt", "")
        prompt = prompt_template.format(metrics_data=data_text)

        response = self.invoke(prompt)
        return self.parser.parse_json(response)

    def build_complete_investigation(
        self,
        hypothesis_id: str,
        hypothesis_description: str,
        insights: List[InsightResult]
    ) -> InsightsEngineOutput:
        """
        Build complete investigation output from all insights.

        Args:
            hypothesis_id: ID of the hypothesis
            hypothesis_description: Description of what was investigated
            insights: All insights gathered

        Returns:
            Complete InsightsEngineOutput
        """
        # Get pattern analysis
        pattern_result = self.analyze_patterns(insights)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            hypothesis_description,
            pattern_result.get("root_cause_hypothesis", ""),
            insights
        )

        # Build investigation path
        investigation_path = [i.hierarchy_level for i in insights]

        # Find highest root cause likelihood
        max_likelihood = max(i.root_cause_likelihood for i in insights) if insights else 0.0

        return InsightsEngineOutput(
            hypothesis_id=hypothesis_id,
            hypothesis_description=hypothesis_description,
            investigation_path=investigation_path,
            insights=insights,
            root_cause_summary=pattern_result.get("root_cause_hypothesis", "Unable to determine"),
            confidence_score=max_likelihood,
            overall_recommendations=[r.get("action", "") for r in recommendations[:5]]
        )
