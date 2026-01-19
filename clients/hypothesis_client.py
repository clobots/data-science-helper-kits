"""Client for hypothesis detection LLM interactions."""

from datetime import date
from typing import List, Optional

from config.settings import settings
from models.llm_inputs import HypothesisInput
from models.llm_outputs import HypothesisEngineOutput, HypothesisResult

from .base_client import BaseGeminiClient


class HypothesisClient(BaseGeminiClient):
    """Client for generating and validating hypotheses using Gemini."""

    def __init__(self, temperature: Optional[float] = None):
        """Initialize with analytical temperature setting."""
        prompts = settings.load_prompt("hypothesis.yaml")
        system_prompt = prompts.get("system_prompt", "")

        super().__init__(
            temperature=temperature or settings.temperature_analytical,
            system_prompt=system_prompt
        )
        self.prompts = prompts

    def detect_hypotheses(
        self,
        input_data: HypothesisInput,
        analysis_date: Optional[date] = None
    ) -> HypothesisEngineOutput:
        """
        Detect significant changes and generate hypotheses.

        Args:
            input_data: HypothesisInput with metric comparisons
            analysis_date: Date of analysis (defaults to today)

        Returns:
            HypothesisEngineOutput with detected hypotheses
        """
        context = input_data.to_prompt_context()
        prompt_template = self.prompts.get("detection_prompt", "")
        prompt = prompt_template.format(context=context)

        # Get raw response and parse
        response = self.invoke(prompt)

        # Parse to output model
        output = self.parser.parse_to_model(
            response,
            HypothesisEngineOutput,
            strict=False
        )

        # Ensure analysis date is set
        if output.analysis_date is None:
            output.analysis_date = analysis_date or date.today()

        return output

    def validate_hypothesis(
        self,
        hypothesis: HypothesisResult,
        supporting_data: dict
    ) -> dict:
        """
        Validate a single hypothesis with additional data.

        Args:
            hypothesis: The hypothesis to validate
            supporting_data: Additional data for validation

        Returns:
            Validation result with confidence and concerns
        """
        prompt_template = self.prompts.get("validation_prompt", "")
        prompt = prompt_template.format(
            hypothesis=hypothesis.description,
            data=str(supporting_data)
        )

        response = self.invoke(prompt)
        return self.parser.parse_json(response)

    def prioritize_hypotheses(
        self,
        hypotheses: List[HypothesisResult]
    ) -> List[dict]:
        """
        Prioritize hypotheses for investigation.

        Args:
            hypotheses: List of hypotheses to prioritize

        Returns:
            Prioritized list with rankings and rationale
        """
        hypotheses_text = "\n".join([
            f"- {h.id}: {h.description} (significance: {h.significance_score})"
            for h in hypotheses
        ])

        prompt_template = self.prompts.get("prioritization_prompt", "")
        prompt = prompt_template.format(hypotheses=hypotheses_text)

        response = self.invoke(prompt)
        result = self.parser.parse_json(response)

        return result.get("prioritized_hypotheses", [])

    def generate_hypothesis_description(
        self,
        metric: str,
        change_type: str,
        change_value: float,
        hierarchy_level: str,
        hierarchy_name: str
    ) -> str:
        """
        Generate a human-readable description for a hypothesis.

        Args:
            metric: The affected metric
            change_type: Type of change detected
            change_value: The magnitude of change
            hierarchy_level: Where the change was detected
            hierarchy_name: Name of the affected entity

        Returns:
            Human-readable hypothesis description
        """
        change_descriptions = {
            "wow_increase": f"increased by {abs(change_value):.1f}%",
            "wow_decrease": f"decreased by {abs(change_value):.1f}%",
            "budget_over": f"is {abs(change_value):.1f}% over budget",
            "budget_under": f"is {abs(change_value):.1f}% under budget",
            "trend_anomaly": f"shows anomalous behavior ({change_value:+.1f}% deviation)"
        }

        change_text = change_descriptions.get(
            change_type,
            f"changed by {change_value:+.1f}%"
        )

        return (
            f"{metric.replace('_', ' ').title()} at {hierarchy_name} ({hierarchy_level}) "
            f"has {change_text} week-over-week"
        )
