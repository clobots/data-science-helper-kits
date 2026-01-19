"""Client for the general summariser tool LLM interactions."""

from typing import Literal, Optional

from config.settings import settings
from models.llm_inputs import SummariserInput
from models.llm_outputs import SummariserOutput

from .base_client import BaseGeminiClient


class SummariserClient(BaseGeminiClient):
    """Client for general-purpose summarisation using Gemini."""

    def __init__(self, temperature: Optional[float] = None):
        """Initialize with creative temperature for better summaries."""
        prompts = settings.load_prompt("summariser.yaml")
        system_prompt = prompts.get("system_prompt", "")

        super().__init__(
            temperature=temperature or settings.temperature_creative,
            system_prompt=system_prompt
        )
        self.prompts = prompts

    def summarise(
        self,
        input_data: SummariserInput
    ) -> SummariserOutput:
        """
        Generate a summary based on input specifications.

        Args:
            input_data: SummariserInput with content and preferences

        Returns:
            SummariserOutput with generated summary
        """
        context = input_data.to_prompt_context()
        prompt_template = self.prompts.get("summarise_prompt", "")

        prompt = prompt_template.format(
            context=context,
            content=input_data.content
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            SummariserOutput,
            strict=False
        )

    def summarise_for_audience(
        self,
        content: str,
        audience_goal: str,
        expertise: Literal["low", "medium", "high"] = "medium",
        tone: Literal["formal", "conversational", "urgent", "encouraging"] = "formal",
        format_type: Literal["paragraph", "bullets", "structured"] = "structured"
    ) -> SummariserOutput:
        """
        Convenience method for common summarisation patterns.

        Args:
            content: Content to summarise
            audience_goal: What the audience needs
            expertise: Audience expertise level
            tone: Desired tone
            format_type: Output format

        Returns:
            SummariserOutput with generated summary
        """
        input_data = SummariserInput(
            content=content,
            audience_goal=audience_goal,
            audience_expertise=expertise,
            tone=tone,
            format=format_type
        )

        return self.summarise(input_data)

    def summarise_with_custom_instructions(
        self,
        content: str,
        custom_instructions: str
    ) -> SummariserOutput:
        """
        Generate a summary with custom instructions.

        Args:
            content: Content to summarise
            custom_instructions: Specific instructions to follow

        Returns:
            SummariserOutput with generated summary
        """
        prompt_template = self.prompts.get("custom_prompt", "")

        prompt = prompt_template.format(
            custom_instructions=custom_instructions,
            content=content
        )

        response = self.invoke(prompt)

        return self.parser.parse_to_model(
            response,
            SummariserOutput,
            strict=False
        )

    def get_tone_guidelines(self, tone: str) -> dict:
        """
        Get guidelines for a specific tone.

        Args:
            tone: The tone to get guidelines for

        Returns:
            Guidelines dictionary with description, examples, and things to avoid
        """
        tone_guidelines = self.prompts.get("tone_guidelines", {})
        return tone_guidelines.get(tone, {})

    def get_expertise_guidelines(self, expertise: str) -> dict:
        """
        Get guidelines for adapting to expertise level.

        Args:
            expertise: The expertise level (low, medium, high)

        Returns:
            Guidelines dictionary with adaptation strategies
        """
        expertise_adaptation = self.prompts.get("expertise_adaptation", {})
        return expertise_adaptation.get(expertise, {})

    def extract_key_points(
        self,
        content: str,
        max_points: int = 5
    ) -> list[str]:
        """
        Extract key points from content without full summarisation.

        Args:
            content: Content to analyze
            max_points: Maximum number of points to extract

        Returns:
            List of key points
        """
        prompt = f"""
Extract the {max_points} most important points from this content.
Return as a JSON array of strings.

Content:
{content}

Return format:
{{"key_points": ["point 1", "point 2", ...]}}
"""

        response = self.invoke(prompt)
        result = self.parser.parse_json(response)

        return result.get("key_points", [])[:max_points]
