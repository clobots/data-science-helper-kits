"""Metric calculation utilities for hypothesis detection and analysis."""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from models.data_models import (
    AggregatedMetric,
    HierarchyLevel,
    MetricType,
    WeeklyComparison,
)


class MetricsCalculator:
    """Calculate metrics for hypothesis detection and analysis."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the calculator with thresholds.

        Args:
            thresholds: Override default significance thresholds
        """
        self.thresholds = thresholds or {
            "wow_change_pct": settings.thresholds.wow_change_pct,
            "budget_variance_pct": settings.thresholds.budget_variance_pct,
            "trend_deviation_pct": settings.thresholds.trend_deviation_pct,
        }

    def calculate_weekly_comparison(
        self,
        df: pd.DataFrame,
        metric: MetricType,
        hierarchy_level: HierarchyLevel,
        current_week: date
    ) -> List[WeeklyComparison]:
        """
        Calculate week-over-week comparisons from a DataFrame.

        Args:
            df: DataFrame with columns: week_ending, hierarchy_id, hierarchy_name, value, budget
            metric: The metric being analyzed
            hierarchy_level: The hierarchy level of the data
            current_week: The current week ending date

        Returns:
            List of WeeklyComparison objects
        """
        previous_week = current_week - timedelta(weeks=1)

        # Get current and previous week data
        current_data = df[df["week_ending"] == current_week].copy()
        previous_data = df[df["week_ending"] == previous_week].copy()

        # Merge on hierarchy
        merged = current_data.merge(
            previous_data[["hierarchy_id", "value"]],
            on="hierarchy_id",
            suffixes=("", "_prev")
        )

        comparisons = []
        for _, row in merged.iterrows():
            comparison = WeeklyComparison.calculate(
                metric_name=metric,
                hierarchy_level=hierarchy_level,
                hierarchy_id=row["hierarchy_id"],
                hierarchy_name=row["hierarchy_name"],
                current_week=current_week,
                current_value=row["value"],
                previous_week_value=row["value_prev"],
                budget_value=row.get("budget")
            )
            comparisons.append(comparison)

        return comparisons

    def calculate_trend(
        self,
        values: List[float],
        dates: List[date]
    ) -> Tuple[str, float, float]:
        """
        Calculate trend direction and strength.

        Args:
            values: List of metric values
            dates: Corresponding dates

        Returns:
            Tuple of (direction, slope, r_squared)
        """
        if len(values) < settings.thresholds.min_sample_size:
            return "insufficient_data", 0.0, 0.0

        # Convert dates to numeric (days from first date)
        first_date = min(dates)
        x = np.array([(d - first_date).days for d in dates])
        y = np.array(values)

        # Linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        sum_y2 = np.sum(y * y)

        # Slope and intercept
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return "stable", 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # R-squared (coefficient of determination)
        ss_tot = sum_y2 - (sum_y ** 2) / n
        y_pred = slope * x + (sum_y - slope * sum_x) / n
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "up"
        else:
            direction = "down"

        return direction, float(slope), float(r_squared)

    def is_significant_wow_change(
        self,
        change_pct: float,
        metric: MetricType
    ) -> bool:
        """
        Determine if a week-over-week change is significant.

        Args:
            change_pct: The percentage change
            metric: The metric type

        Returns:
            True if the change is significant
        """
        threshold = self.thresholds["wow_change_pct"]

        # Some metrics may have different thresholds
        metric_thresholds = {
            MetricType.SALES: threshold * 1.5,  # Sales more volatile
            MetricType.ORDER_PICKRATE: threshold * 0.8,  # Pickrate more sensitive
        }

        effective_threshold = metric_thresholds.get(metric, threshold)
        return abs(change_pct) >= effective_threshold

    def is_significant_budget_variance(
        self,
        variance_pct: float,
        metric: MetricType
    ) -> bool:
        """
        Determine if a budget variance is significant.

        Args:
            variance_pct: The percentage variance from budget
            metric: The metric type

        Returns:
            True if the variance is significant
        """
        threshold = self.thresholds["budget_variance_pct"]
        return abs(variance_pct) >= threshold

    def is_significant_trend_deviation(
        self,
        current_value: float,
        trend_value: float
    ) -> bool:
        """
        Determine if current value deviates significantly from trend.

        Args:
            current_value: The current metric value
            trend_value: The expected value based on trend

        Returns:
            True if there's significant deviation
        """
        if trend_value == 0:
            return current_value != 0

        deviation_pct = abs((current_value - trend_value) / trend_value * 100)
        return deviation_pct >= self.thresholds["trend_deviation_pct"]

    def calculate_significance_score(
        self,
        wow_change_pct: float,
        budget_variance_pct: Optional[float],
        trend_deviation_pct: Optional[float]
    ) -> float:
        """
        Calculate an overall significance score (0-1).

        Args:
            wow_change_pct: Week-over-week change percentage
            budget_variance_pct: Budget variance percentage
            trend_deviation_pct: Trend deviation percentage

        Returns:
            Significance score between 0 and 1
        """
        scores = []

        # WoW score (normalized to 0-1)
        wow_score = min(abs(wow_change_pct) / (self.thresholds["wow_change_pct"] * 3), 1.0)
        scores.append(wow_score * 0.4)  # 40% weight

        # Budget variance score
        if budget_variance_pct is not None:
            budget_score = min(abs(budget_variance_pct) / (self.thresholds["budget_variance_pct"] * 3), 1.0)
            scores.append(budget_score * 0.35)  # 35% weight
        else:
            # Redistribute weight if no budget
            scores[0] = wow_score * 0.6

        # Trend deviation score
        if trend_deviation_pct is not None:
            trend_score = min(abs(trend_deviation_pct) / (self.thresholds["trend_deviation_pct"] * 3), 1.0)
            scores.append(trend_score * 0.25)  # 25% weight

        return min(sum(scores), 1.0)

    def aggregate_to_level(
        self,
        df: pd.DataFrame,
        target_level: HierarchyLevel,
        hierarchy_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate data to a higher hierarchy level.

        Args:
            df: Source DataFrame
            target_level: Target hierarchy level
            hierarchy_mapping: Mapping from current level IDs to target level IDs

        Returns:
            Aggregated DataFrame
        """
        if hierarchy_mapping:
            df = df.copy()
            df["target_id"] = df["hierarchy_id"].map(hierarchy_mapping)
            group_col = "target_id"
        else:
            group_col = "hierarchy_id"

        agg_df = df.groupby(["week_ending", group_col]).agg({
            "value": "mean",
            "budget": "mean"
        }).reset_index()

        agg_df["hierarchy_level"] = target_level.value
        agg_df.rename(columns={group_col: "hierarchy_id"}, inplace=True)

        return agg_df

    def identify_outliers(
        self,
        values: List[float],
        method: str = "iqr"
    ) -> List[int]:
        """
        Identify outlier indices in a list of values.

        Args:
            values: List of values to analyze
            method: "iqr" for interquartile range, "zscore" for z-score

        Returns:
            List of indices that are outliers
        """
        arr = np.array(values)
        outlier_indices = []

        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for i, val in enumerate(values):
                if val < lower_bound or val > upper_bound:
                    outlier_indices.append(i)

        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 0:
                z_scores = (arr - mean) / std
                for i, z in enumerate(z_scores):
                    if abs(z) > 3:
                        outlier_indices.append(i)

        return outlier_indices

    def calculate_contribution(
        self,
        part_value: float,
        total_value: float
    ) -> float:
        """
        Calculate what percentage a part contributes to total.

        Args:
            part_value: The part's value
            total_value: The total value

        Returns:
            Contribution percentage
        """
        if total_value == 0:
            return 0.0
        return (part_value / total_value) * 100
