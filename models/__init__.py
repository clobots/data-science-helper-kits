"""Pydantic models for data structures and LLM interactions."""

from .data_models import (
    MetricData,
    WeeklyComparison,
    HierarchyNode,
    AggregatedMetric,
)
from .llm_inputs import (
    HypothesisInput,
    InsightsInput,
    SummaryInput,
    SummariserInput,
)
from .llm_outputs import (
    HypothesisResult,
    InsightResult,
    SummaryOutput,
    SummariserOutput,
)

__all__ = [
    # Data models
    "MetricData",
    "WeeklyComparison",
    "HierarchyNode",
    "AggregatedMetric",
    # LLM inputs
    "HypothesisInput",
    "InsightsInput",
    "SummaryInput",
    "SummariserInput",
    # LLM outputs
    "HypothesisResult",
    "InsightResult",
    "SummaryOutput",
    "SummariserOutput",
]
