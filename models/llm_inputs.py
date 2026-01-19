"""Pydantic models for LLM input structures."""

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .data_models import HierarchyLevel, MetricType, WeeklyComparison


class HypothesisInput(BaseModel):
    """Input model for hypothesis generation."""

    metric_comparisons: List[WeeklyComparison] = Field(
        description="Weekly comparison data for analysis"
    )
    analysis_period: Dict[str, date] = Field(
        description="Start and end dates for analysis"
    )
    significance_thresholds: Dict[str, float] = Field(
        description="Thresholds for determining significance"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional business context for analysis"
    )

    class Config:
        use_enum_values = True

    def to_prompt_context(self) -> str:
        """Convert to formatted string for prompt injection."""
        lines = [
            f"Analysis Period: {self.analysis_period.get('start')} to {self.analysis_period.get('end')}",
            f"Significance Thresholds:",
            f"  - Week-over-Week Change: {self.significance_thresholds.get('wow_change_pct', 5.0)}%",
            f"  - Budget Variance: {self.significance_thresholds.get('budget_variance_pct', 3.0)}%",
            "",
            "Metric Comparisons:",
        ]

        for comp in self.metric_comparisons[:20]:  # Limit for context length
            lines.append(
                f"  - {comp.metric_name} at {comp.hierarchy_name}: "
                f"Current={comp.current_value:.2f}, WoW Change={comp.wow_change_pct:+.1f}%, "
                f"Budget Variance={comp.budget_variance_pct:+.1f}%" if comp.budget_variance_pct else ""
            )

        if self.context:
            lines.extend(["", f"Additional Context: {self.context}"])

        return "\n".join(lines)


class InsightsInput(BaseModel):
    """Input model for insights generation."""

    hypothesis_description: str = Field(
        description="The hypothesis being investigated"
    )
    metric_name: MetricType = Field(
        description="The metric being analyzed"
    )
    hierarchy_level: HierarchyLevel = Field(
        description="Current hierarchy level of investigation"
    )
    hierarchy_data: List[Dict] = Field(
        description="Data at the current hierarchy level"
    )
    parent_insight: Optional[str] = Field(
        default=None,
        description="Insight from parent level for context"
    )
    drill_down_path: List[str] = Field(
        default_factory=list,
        description="Path of hierarchy levels already investigated"
    )

    class Config:
        use_enum_values = True

    def to_prompt_context(self) -> str:
        """Convert to formatted string for prompt injection."""
        lines = [
            f"Hypothesis: {self.hypothesis_description}",
            f"Metric: {self.metric_name}",
            f"Current Level: {self.hierarchy_level}",
        ]

        if self.drill_down_path:
            lines.append(f"Investigation Path: {' -> '.join(self.drill_down_path)}")

        if self.parent_insight:
            lines.append(f"Parent Level Insight: {self.parent_insight}")

        lines.extend(["", "Data at Current Level:"])
        for item in self.hierarchy_data[:15]:  # Limit for context
            lines.append(f"  - {item}")

        return "\n".join(lines)


class SummaryInput(BaseModel):
    """Input model for the summarisation engine."""

    insights: List[Dict] = Field(
        description="All insights to be summarised"
    )
    hypotheses: List[Dict] = Field(
        description="Original hypotheses that were investigated"
    )
    audience: Literal["executive", "analyst", "operations"] = Field(
        default="analyst",
        description="Target audience for the summary"
    )
    tone: Literal["formal", "conversational", "urgent"] = Field(
        default="formal",
        description="Tone for the summary"
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length of summary in words"
    )

    def to_prompt_context(self) -> str:
        """Convert to formatted string for prompt injection."""
        lines = [
            f"Audience: {self.audience}",
            f"Tone: {self.tone}",
        ]

        if self.max_length:
            lines.append(f"Maximum Length: {self.max_length} words")

        lines.extend(["", "Hypotheses Investigated:"])
        for idx, hyp in enumerate(self.hypotheses, 1):
            lines.append(f"  {idx}. {hyp.get('description', hyp)}")

        lines.extend(["", "Insights Generated:"])
        for idx, insight in enumerate(self.insights, 1):
            if isinstance(insight, dict):
                lines.append(f"  {idx}. {insight.get('summary', insight)}")
            else:
                lines.append(f"  {idx}. {insight}")

        return "\n".join(lines)


class SummariserInput(BaseModel):
    """Input model for the general summariser tool."""

    content: str = Field(
        description="Content to be summarised"
    )
    audience_goal: str = Field(
        description="What the audience needs to understand or do"
    )
    audience_expertise: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Technical expertise level of audience"
    )
    tone: Literal["formal", "conversational", "urgent", "encouraging"] = Field(
        default="formal",
        description="Desired tone for the summary"
    )
    format: Literal["paragraph", "bullets", "structured"] = Field(
        default="structured",
        description="Output format preference"
    )
    key_points_limit: Optional[int] = Field(
        default=5,
        description="Maximum number of key points to include"
    )

    def to_prompt_context(self) -> str:
        """Convert to formatted string for prompt injection."""
        lines = [
            f"Audience Goal: {self.audience_goal}",
            f"Audience Expertise: {self.audience_expertise}",
            f"Tone: {self.tone}",
            f"Format: {self.format}",
        ]

        if self.key_points_limit:
            lines.append(f"Key Points Limit: {self.key_points_limit}")

        lines.extend(["", "Content to Summarise:", self.content])

        return "\n".join(lines)


class DataExtractorInput(BaseModel):
    """Input model for data extraction queries."""

    query_type: Literal["raw", "aggregated", "comparison", "trend"] = Field(
        description="Type of query to execute"
    )
    metrics: List[MetricType] = Field(
        description="Metrics to extract"
    )
    hierarchy_level: HierarchyLevel = Field(
        description="Level of aggregation"
    )
    hierarchy_filter: Optional[List[str]] = Field(
        default=None,
        description="Specific hierarchy IDs to filter"
    )
    date_range: Dict[str, date] = Field(
        description="Start and end dates for data"
    )
    include_budget: bool = Field(
        default=True,
        description="Include budget values"
    )
    include_previous_period: bool = Field(
        default=True,
        description="Include previous period for comparison"
    )

    class Config:
        use_enum_values = True
