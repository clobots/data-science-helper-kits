"""Client for summary generation LLM interactions."""

from datetime import date
from typing import Dict, List, Literal, Optional

from config.settings import settings
from models.llm_inputs import SummaryInput
from models.llm_outputs import SummaryOutput

from .base_client import BaseGeminiClient


class SummaryClient(BaseGeminiClient):
    """Client for generating audience-specific summaries using Gemini."""

    def __init__(self, temperature: Optional[float] = None):
        """Initialize with creative temperature for better narratives."""
        prompts = settings.load_prompt("summary.yaml")
        system_prompt = prompts.get("system_prompt", "")

        super().__init__(
            temperature=temperature or settings.temperature_creative,
            system_prompt=system_prompt
        )
        self.prompts = prompts

    def generate_executive_summary(
        self,
        hypotheses: List[Dict],
        insights: List[Dict],
        tone: Literal["formal", "conversational", "urgent"] = "formal"
    ) -> SummaryOutput:
        """
        Generate an executive-level summary.

        Args:
            hypotheses: Investigated hypotheses
            insights: Generated insights
            tone: Desired tone for the summary

        Returns:
            SummaryOutput tailored for executives
        """
        hypotheses_text = "\n".join([
            f"- {h.get('description', h)}"
            for h in hypotheses
        ])

        insights_text = "\n".join([
            f"- {i.get('finding', i.get('summary', i))}"
            for i in insights
        ])

        prompt_template = self.prompts.get("executive_summary_prompt", "")
        prompt = prompt_template.format(
            hypotheses=hypotheses_text,
            insights=insights_text,
            tone=tone,
            date=date.today().isoformat()
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            SummaryOutput,
            strict=False
        )

    def generate_analyst_summary(
        self,
        overview: str,
        hypotheses: List[Dict],
        insights: List[Dict],
        methodology: str
    ) -> SummaryOutput:
        """
        Generate a detailed analyst-level summary.

        Args:
            overview: High-level analysis overview
            hypotheses: Investigated hypotheses with results
            insights: Detailed insights
            methodology: Description of methodology used

        Returns:
            SummaryOutput with detailed analysis
        """
        prompt_template = self.prompts.get("analyst_summary_prompt", "")
        prompt = prompt_template.format(
            overview=overview,
            hypotheses=str(hypotheses),
            insights=str(insights),
            methodology=methodology,
            date=date.today().isoformat()
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            SummaryOutput,
            strict=False
        )

    def generate_operations_summary(
        self,
        issues: List[Dict],
        affected_areas: List[str],
        recommendations: List[Dict],
        tone: Literal["formal", "conversational", "urgent"] = "formal"
    ) -> SummaryOutput:
        """
        Generate an operations-focused summary.

        Args:
            issues: Identified issues
            affected_areas: List of affected stores/categories
            recommendations: Action recommendations
            tone: Desired tone

        Returns:
            SummaryOutput with operational focus
        """
        prompt_template = self.prompts.get("operations_summary_prompt", "")
        prompt = prompt_template.format(
            issues=str(issues),
            affected_areas=", ".join(affected_areas),
            recommendations=str(recommendations),
            tone=tone,
            date=date.today().isoformat()
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            SummaryOutput,
            strict=False
        )

    def generate_summary(
        self,
        input_data: SummaryInput
    ) -> SummaryOutput:
        """
        Generate a summary based on input specifications.

        Args:
            input_data: SummaryInput with all parameters

        Returns:
            SummaryOutput matching the audience and tone
        """
        if input_data.audience == "executive":
            return self.generate_executive_summary(
                input_data.hypotheses,
                input_data.insights,
                input_data.tone
            )
        elif input_data.audience == "analyst":
            return self.generate_analyst_summary(
                overview="Analysis of Woolworths operational metrics",
                hypotheses=input_data.hypotheses,
                insights=input_data.insights,
                methodology="Statistical analysis of WoW changes and budget variances"
            )
        else:  # operations
            return self.generate_operations_summary(
                issues=input_data.hypotheses,
                affected_areas=[i.get("hierarchy_name", "Unknown") for i in input_data.insights],
                recommendations=[{"action": i.get("finding", "")} for i in input_data.insights],
                tone=input_data.tone
            )

    def generate_alert(
        self,
        issues: List[Dict],
        impact: Dict
    ) -> Dict:
        """
        Generate an urgent alert summary.

        Args:
            issues: Critical issues identified
            impact: Impact assessment

        Returns:
            Alert summary with severity and actions
        """
        prompt_template = self.prompts.get("alert_summary_prompt", "")
        prompt = prompt_template.format(
            issues=str(issues),
            impact=str(impact)
        )

        response = self.invoke(prompt)
        return self.parser.parse_json(response)

    def generate_multi_audience(
        self,
        results: Dict
    ) -> Dict:
        """
        Generate summaries for multiple audiences.

        Args:
            results: Complete analysis results

        Returns:
            Dictionary with summaries for each audience
        """
        prompt_template = self.prompts.get("multi_audience_prompt", "")
        prompt = prompt_template.format(results=str(results))

        response = self.invoke(prompt)
        return self.parser.parse_json(response)
