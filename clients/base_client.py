"""Base client for Gemini LLM interactions using LangChain."""

from typing import Any, Dict, Optional, Type, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from config.settings import settings
from utils.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class BaseGeminiClient:
    """Base client for interacting with Google Gemini via LangChain."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Gemini client.

        Args:
            temperature: Temperature for generation (0.0-1.0)
            model_name: Override default model name
            system_prompt: System prompt for the client
        """
        self.model_name = model_name or settings.model_name
        self.temperature = temperature or settings.temperature_analytical
        self.system_prompt = system_prompt
        self.parser = ResponseParser()

        self._llm = self._create_llm()

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the LangChain Gemini client."""
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Please set it in your environment or .env file."
            )

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.gemini_api_key,
            temperature=self.temperature,
            convert_system_message_to_human=True,
        )

    def _build_messages(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> list:
        """Build message list for the LLM."""
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        user_content = prompt
        if context:
            user_content = f"{context}\n\n{prompt}"

        messages.append(HumanMessage(content=user_content))

        return messages

    def invoke(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: The main prompt
            context: Optional context to prepend

        Returns:
            Raw string response from the LLM
        """
        messages = self._build_messages(prompt, context)
        response = self._llm.invoke(messages)
        return response.content

    def invoke_with_schema(
        self,
        prompt: str,
        output_schema: Type[T],
        context: Optional[str] = None,
        strict: bool = True
    ) -> T:
        """
        Invoke the LLM and parse response to a Pydantic model.

        Args:
            prompt: The main prompt
            output_schema: Pydantic model class for the response
            context: Optional context to prepend
            strict: Whether to raise on parsing errors

        Returns:
            Parsed Pydantic model instance
        """
        # Add schema instructions to prompt
        schema_prompt = self._add_schema_instructions(prompt, output_schema)

        # Get raw response
        raw_response = self.invoke(schema_prompt, context)

        # Parse to Pydantic model
        return self.parser.parse_to_model(raw_response, output_schema, strict=strict)

    def _add_schema_instructions(
        self,
        prompt: str,
        output_schema: Type[BaseModel]
    ) -> str:
        """Add JSON schema instructions to the prompt."""
        schema_json = output_schema.model_json_schema()

        schema_instruction = f"""
Please respond with a valid JSON object that matches this schema:

```json
{schema_json}
```

Important:
- Return ONLY the JSON object, no additional text
- Ensure all required fields are present
- Use the exact field names from the schema
- For enum fields, use only the allowed values

"""
        return schema_instruction + prompt

    def invoke_batch(
        self,
        prompts: list[str],
        context: Optional[str] = None
    ) -> list[str]:
        """
        Invoke the LLM with multiple prompts.

        Args:
            prompts: List of prompts
            context: Optional shared context

        Returns:
            List of string responses
        """
        results = []
        for prompt in prompts:
            result = self.invoke(prompt, context)
            results.append(result)
        return results

    def get_token_estimate(self, text: str) -> int:
        """
        Estimate token count for a text.

        Note: This is a rough estimate. Actual tokenization may vary.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    def set_temperature(self, temperature: float) -> None:
        """Update the temperature setting."""
        self.temperature = temperature
        self._llm = self._create_llm()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = system_prompt
