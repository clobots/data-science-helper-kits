"""Configuration settings for the data science agent."""

import os
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class SignificanceThresholds(BaseModel):
    """Thresholds for detecting significant metric changes."""

    wow_change_pct: float = Field(default=5.0, description="Week-over-week change threshold (%)")
    budget_variance_pct: float = Field(default=3.0, description="Budget variance threshold (%)")
    trend_deviation_pct: float = Field(default=10.0, description="Trend deviation threshold (%)")
    min_sample_size: int = Field(default=4, description="Minimum weeks for trend analysis")


class Settings(BaseModel):
    """Main settings for the data science agent."""

    # API Configuration
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API key"
    )
    model_name: str = Field(
        default="gemini-2.5-flash-lite",
        description="Gemini model to use"
    )

    # Temperature settings for different use cases
    temperature_analytical: float = Field(default=0.1, description="Temperature for analytical tasks")
    temperature_creative: float = Field(default=0.7, description="Temperature for summaries")

    # Significance thresholds
    thresholds: SignificanceThresholds = Field(default_factory=SignificanceThresholds)

    # Hierarchy levels (from granular to broad)
    hierarchy_levels: List[str] = Field(
        default=["category", "store", "group", "state", "zone", "national"],
        description="Data hierarchy levels"
    )

    # Metrics tracked
    metrics: List[str] = Field(
        default=["wages", "voice_of_customer", "sales", "stockloss", "order_pickrate"],
        description="Metrics to analyze"
    )

    # Metric display names and descriptions
    metric_descriptions: Dict[str, str] = Field(
        default={
            "wages": "Labor cost as percentage of sales",
            "voice_of_customer": "Customer satisfaction score (0-100)",
            "sales": "Total sales in dollars",
            "stockloss": "Inventory shrinkage as percentage of sales",
            "order_pickrate": "Online order fulfillment rate (items per hour)"
        }
    )

    # Data generation settings
    num_weeks: int = Field(default=12, description="Weeks of historical data")
    num_stores: int = Field(default=100, description="Number of stores (reduced for demo)")
    num_categories: int = Field(default=20, description="Number of product categories")

    # Paths
    base_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base project path"
    )

    @property
    def prompts_path(self) -> Path:
        """Path to prompt templates."""
        return self.base_path / "config" / "prompts"

    @property
    def data_path(self) -> Path:
        """Path to sample data."""
        return self.base_path / "data" / "sample_data"

    def load_prompt(self, prompt_file: str) -> Dict:
        """Load a prompt template from YAML file."""
        prompt_path = self.prompts_path / prompt_file
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# Global settings instance
settings = Settings()
