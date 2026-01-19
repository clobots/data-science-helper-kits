"""Parse LLM responses to Pydantic models."""

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ResponseParseError(Exception):
    """Error parsing LLM response."""

    def __init__(self, message: str, raw_response: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.details = details or {}


class ResponseParser:
    """Parse LLM string responses into Pydantic models."""

    # Patterns for extracting JSON from various formats
    JSON_PATTERNS = [
        r"```json\s*([\s\S]*?)\s*```",  # Markdown code block
        r"```\s*([\s\S]*?)\s*```",       # Generic code block
        r"\{[\s\S]*\}",                   # Raw JSON object
        r"\[[\s\S]*\]",                   # Raw JSON array
    ]

    def extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM response text.

        Args:
            text: Raw LLM response

        Returns:
            Extracted JSON string

        Raises:
            ResponseParseError: If no valid JSON found
        """
        # Clean up common issues
        text = text.strip()

        # Try each pattern
        for pattern in self.JSON_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                # Take the first match
                json_str = matches[0] if isinstance(matches[0], str) else matches[0]
                # Validate it's parseable
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue

        # If no patterns matched, try the whole text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        raise ResponseParseError(
            "Could not extract valid JSON from response",
            raw_response=text
        )

    def parse_json(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from text.

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON as dictionary
        """
        json_str = self.extract_json(text)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ResponseParseError(
                f"JSON parse error: {e}",
                raw_response=text,
                details={"error": str(e)}
            )

    def parse_to_model(
        self,
        text: str,
        model_class: Type[T],
        strict: bool = True
    ) -> T:
        """
        Parse LLM response directly to a Pydantic model.

        Args:
            text: Raw LLM response
            model_class: Target Pydantic model class
            strict: If True, raise on validation errors; if False, attempt partial parsing

        Returns:
            Pydantic model instance
        """
        try:
            data = self.parse_json(text)
        except ResponseParseError:
            if strict:
                raise
            # Attempt to create a minimal valid model
            data = {}

        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            if strict:
                raise ResponseParseError(
                    f"Validation error: {e}",
                    raw_response=text,
                    details={"validation_errors": e.errors()}
                )

            # Try to fix common issues
            fixed_data = self._attempt_fix(data, model_class, e)
            return model_class.model_validate(fixed_data)

    def _attempt_fix(
        self,
        data: Dict[str, Any],
        model_class: Type[BaseModel],
        error: ValidationError
    ) -> Dict[str, Any]:
        """
        Attempt to fix common validation issues.

        Args:
            data: Parsed data with issues
            model_class: Target model class
            error: The validation error

        Returns:
            Fixed data dictionary
        """
        fixed = data.copy()
        schema = model_class.model_json_schema()

        for err in error.errors():
            field = err["loc"][0] if err["loc"] else None

            if err["type"] == "missing" and field:
                # Add default for missing required field
                if field in schema.get("properties", {}):
                    prop = schema["properties"][field]
                    if prop.get("type") == "string":
                        fixed[field] = ""
                    elif prop.get("type") == "array":
                        fixed[field] = []
                    elif prop.get("type") == "object":
                        fixed[field] = {}
                    elif prop.get("type") == "number":
                        fixed[field] = 0.0
                    elif prop.get("type") == "integer":
                        fixed[field] = 0
                    elif prop.get("type") == "boolean":
                        fixed[field] = False

            elif err["type"] == "enum" and field:
                # Try to map to a valid enum value
                value = str(data.get(field, "")).lower()
                prop = schema.get("properties", {}).get(field, {})
                enum_values = prop.get("enum", [])
                if enum_values:
                    # Try to find a close match
                    for enum_val in enum_values:
                        if value in str(enum_val).lower():
                            fixed[field] = enum_val
                            break
                    else:
                        # Use first enum value as default
                        fixed[field] = enum_values[0]

        return fixed

    def extract_list(self, text: str, item_class: Type[T]) -> list[T]:
        """
        Parse LLM response containing a list of items.

        Args:
            text: Raw LLM response
            item_class: Pydantic model class for each item

        Returns:
            List of Pydantic model instances
        """
        data = self.parse_json(text)

        # Handle both array and object with array field
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Find the first list field
            for value in data.values():
                if isinstance(value, list):
                    items = value
                    break
            else:
                items = [data]
        else:
            items = []

        return [item_class.model_validate(item) for item in items]

    def safe_parse(
        self,
        text: str,
        model_class: Type[T],
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Safely parse without raising exceptions.

        Args:
            text: Raw LLM response
            model_class: Target model class
            default: Default value if parsing fails

        Returns:
            Parsed model or default
        """
        try:
            return self.parse_to_model(text, model_class, strict=False)
        except Exception:
            return default
