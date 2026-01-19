"""Pydantic models for LLM output structures."""

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .data_models import HierarchyLevel, MetricType


class HypothesisResult(BaseModel):
    """Output model for a detected hypothesis."""

    id: str = Field(description="Unique identifier for the hypothesis")
    metric: MetricType = Field(description="The metric with significant change")
    hierarchy_level: HierarchyLevel = Field(description="Level where change detected")
    hierarchy_id: str = Field(description="ID of the hierarchy node")
    hierarchy_name: str = Field(description="Name of the hierarchy node")

    change_type: Literal["wow_increase", "wow_decrease", "budget_over", "budget_under", "trend_anomaly"] = Field(
        description="Type of significant change detected"
    )
    change_value: float = Field(description="The actual change value")
    change_percentage: float = Field(description="Change as percentage")

    significance_score: float = Field(
        ge=0.0, le=1.0,
        description="How significant the change is (0-1)"
    )
    description: str = Field(description="Human-readable description of the hypothesis")
    requires_investigation: bool = Field(
        default=True,
        description="Whether this needs further investigation"
    )

    class Config:
        use_enum_values = True


class HypothesisEngineOutput(BaseModel):
    """Complete output from the hypothesis engine."""

    analysis_date: date = Field(description="Date of analysis")
    total_comparisons_analyzed: int = Field(description="Number of data points analyzed")
    hypotheses: List[HypothesisResult] = Field(description="All detected hypotheses")
    summary: str = Field(description="Summary of findings")

    @property
    def high_priority_hypotheses(self) -> List[HypothesisResult]:
        """Get hypotheses with significance > 0.7."""
        return [h for h in self.hypotheses if h.significance_score > 0.7]


class InsightResult(BaseModel):
    """Output model for an insight at a hierarchy level."""

    hypothesis_id: str = Field(description="ID of the hypothesis being investigated")
    hierarchy_level: HierarchyLevel = Field(description="Level of this insight")
    hierarchy_id: str = Field(description="ID of the hierarchy node")
    hierarchy_name: str = Field(description="Name of the hierarchy node")

    finding: str = Field(description="The key finding at this level")
    contribution_percentage: float = Field(
        description="How much this level contributes to the overall issue"
    )
    root_cause_likelihood: float = Field(
        ge=0.0, le=1.0,
        description="Likelihood this is a root cause"
    )

    supporting_data: Dict = Field(
        default_factory=dict,
        description="Data supporting this insight"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions based on this insight"
    )
    requires_deeper_investigation: bool = Field(
        default=False,
        description="Whether to drill down further"
    )

    class Config:
        use_enum_values = True


class InsightsEngineOutput(BaseModel):
    """Complete output from insights engine for one hypothesis."""

    hypothesis_id: str = Field(description="ID of the investigated hypothesis")
    hypothesis_description: str = Field(description="Description of what was investigated")
    investigation_path: List[str] = Field(
        description="Path of hierarchy levels investigated"
    )
    insights: List[InsightResult] = Field(
        description="Insights at each level"
    )
    root_cause_summary: str = Field(
        description="Summary of the most likely root cause"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the root cause analysis"
    )
    overall_recommendations: List[str] = Field(
        description="Top recommendations across all levels"
    )


class KeyFinding(BaseModel):
    """A key finding for the summary."""

    title: str = Field(description="Short title for the finding")
    description: str = Field(description="Detailed description")
    impact: Literal["high", "medium", "low"] = Field(description="Impact level")
    affected_areas: List[str] = Field(description="Areas affected by this finding")


class ActionItem(BaseModel):
    """An action item from the analysis."""

    action: str = Field(description="The recommended action")
    priority: Literal["immediate", "short_term", "long_term"] = Field(
        description="Priority level"
    )
    owner: Optional[str] = Field(default=None, description="Suggested owner")
    expected_impact: str = Field(description="Expected impact of taking this action")


class SummaryOutput(BaseModel):
    """Output model for the summarisation engine."""

    audience: str = Field(description="Target audience")
    tone: str = Field(description="Tone used")
    generated_date: date = Field(description="Date summary was generated")

    executive_summary: str = Field(
        description="High-level summary for quick reading"
    )
    key_findings: List[KeyFinding] = Field(
        description="Key findings from the analysis"
    )
    action_items: List[ActionItem] = Field(
        description="Recommended actions"
    )
    detailed_analysis: Optional[str] = Field(
        default=None,
        description="Detailed analysis for analysts"
    )
    appendix: Optional[Dict] = Field(
        default=None,
        description="Supporting data and methodology"
    )


class SummariserOutput(BaseModel):
    """Output model for the general summariser tool."""

    summary: str = Field(description="The generated summary")
    key_points: List[str] = Field(description="Key points extracted")
    tone_applied: str = Field(description="The tone that was applied")
    word_count: int = Field(description="Word count of the summary")
    audience_appropriate: bool = Field(
        description="Whether the summary is appropriate for the audience"
    )
