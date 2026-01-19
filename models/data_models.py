"""Pydantic models for data structures used throughout the agent."""

from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HierarchyLevel(str, Enum):
    """Hierarchy levels in the Woolworths organization."""

    CATEGORY = "category"
    STORE = "store"
    GROUP = "group"
    STATE = "state"
    ZONE = "zone"
    NATIONAL = "national"


class MetricType(str, Enum):
    """Types of metrics tracked."""

    WAGES = "wages"
    VOICE_OF_CUSTOMER = "voice_of_customer"
    SALES = "sales"
    STOCKLOSS = "stockloss"
    ORDER_PICKRATE = "order_pickrate"


class HierarchyNode(BaseModel):
    """Represents a node in the organizational hierarchy."""

    level: HierarchyLevel
    id: str
    name: str
    parent_id: Optional[str] = None

    class Config:
        use_enum_values = True


class MetricData(BaseModel):
    """Individual metric data point."""

    metric_name: MetricType
    value: float
    week_ending: date
    hierarchy_level: HierarchyLevel
    hierarchy_id: str
    hierarchy_name: str
    budget: Optional[float] = None
    previous_week_value: Optional[float] = None

    class Config:
        use_enum_values = True


class WeeklyComparison(BaseModel):
    """Comparison data for week-over-week analysis."""

    metric_name: MetricType
    hierarchy_level: HierarchyLevel
    hierarchy_id: str
    hierarchy_name: str
    current_week: date
    current_value: float
    previous_week_value: float
    budget_value: Optional[float] = None

    # Calculated fields
    wow_change: float = Field(description="Week-over-week change amount")
    wow_change_pct: float = Field(description="Week-over-week change percentage")
    budget_variance: Optional[float] = Field(default=None, description="Variance from budget")
    budget_variance_pct: Optional[float] = Field(default=None, description="Variance from budget (%)")

    class Config:
        use_enum_values = True

    @classmethod
    def calculate(
        cls,
        metric_name: MetricType,
        hierarchy_level: HierarchyLevel,
        hierarchy_id: str,
        hierarchy_name: str,
        current_week: date,
        current_value: float,
        previous_week_value: float,
        budget_value: Optional[float] = None
    ) -> "WeeklyComparison":
        """Calculate comparison metrics."""
        wow_change = current_value - previous_week_value
        wow_change_pct = (wow_change / previous_week_value * 100) if previous_week_value != 0 else 0.0

        budget_variance = None
        budget_variance_pct = None
        if budget_value is not None:
            budget_variance = current_value - budget_value
            budget_variance_pct = (budget_variance / budget_value * 100) if budget_value != 0 else 0.0

        return cls(
            metric_name=metric_name,
            hierarchy_level=hierarchy_level,
            hierarchy_id=hierarchy_id,
            hierarchy_name=hierarchy_name,
            current_week=current_week,
            current_value=current_value,
            previous_week_value=previous_week_value,
            budget_value=budget_value,
            wow_change=wow_change,
            wow_change_pct=wow_change_pct,
            budget_variance=budget_variance,
            budget_variance_pct=budget_variance_pct
        )


class AggregatedMetric(BaseModel):
    """Aggregated metric data across time periods."""

    metric_name: MetricType
    hierarchy_level: HierarchyLevel
    hierarchy_id: str
    hierarchy_name: str

    # Time series data
    weekly_values: List[float]
    week_dates: List[date]

    # Calculated statistics
    mean: float
    std_dev: float
    trend_direction: str = Field(description="up, down, or stable")
    trend_strength: float = Field(description="Correlation coefficient of trend")

    class Config:
        use_enum_values = True


class DataQuery(BaseModel):
    """Query parameters for data extraction."""

    metrics: List[MetricType]
    hierarchy_level: HierarchyLevel
    hierarchy_ids: Optional[List[str]] = None
    start_date: date
    end_date: date
    include_budget: bool = True
    aggregate_by: Optional[str] = None

    class Config:
        use_enum_values = True


class QueryResult(BaseModel):
    """Result from a data extraction query."""

    query: DataQuery
    data: List[MetricData]
    row_count: int
    execution_time_ms: float

    class Config:
        use_enum_values = True
