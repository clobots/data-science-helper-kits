"""Data extraction tool for querying simulated BigQuery data."""

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from data.generator import WoolworthsDataGenerator
from models.data_models import (
    DataQuery,
    HierarchyLevel,
    MetricData,
    MetricType,
    QueryResult,
    WeeklyComparison,
)
from utils.metrics import MetricsCalculator


class DataExtractor:
    """
    Simulated BigQuery data extraction tool.

    In production, this would connect to BigQuery and execute SQL queries.
    For demo purposes, it uses locally generated CSV data.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data extractor.

        Args:
            data_path: Path to data files (uses settings default if not provided)
        """
        self.data_path = data_path or settings.data_path
        self.metrics_calculator = MetricsCalculator()
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._generator: Optional[WoolworthsDataGenerator] = None

    def _ensure_data_exists(self) -> None:
        """Ensure sample data files exist, generate if needed."""
        if not self.data_path.exists():
            self._generator = WoolworthsDataGenerator(seed=42)
            self._generator.save_data(self.data_path)
        elif not any(self.data_path.glob("*.csv")):
            self._generator = WoolworthsDataGenerator(seed=42)
            self._generator.save_data(self.data_path)

    def _load_metric_data(self, metric: MetricType) -> pd.DataFrame:
        """Load data for a specific metric."""
        if metric.value in self._data_cache:
            return self._data_cache[metric.value]

        self._ensure_data_exists()
        file_path = self.data_path / f"{metric.value}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["week_ending"])
        self._data_cache[metric.value] = df

        return df

    def _load_hierarchy(self) -> pd.DataFrame:
        """Load hierarchy data."""
        if "hierarchy" in self._data_cache:
            return self._data_cache["hierarchy"]

        self._ensure_data_exists()
        file_path = self.data_path / "hierarchy.csv"

        if file_path.exists():
            df = pd.read_csv(file_path)
            self._data_cache["hierarchy"] = df
            return df

        return pd.DataFrame()

    def execute_query(self, query: DataQuery) -> QueryResult:
        """
        Execute a data query.

        Args:
            query: DataQuery specifying what data to retrieve

        Returns:
            QueryResult with the requested data
        """
        start_time = time.time()
        all_data = []

        for metric in query.metrics:
            df = self._load_metric_data(metric)

            # Filter by date range
            df = df[
                (df["week_ending"] >= pd.Timestamp(query.start_date)) &
                (df["week_ending"] <= pd.Timestamp(query.end_date))
            ]

            # Filter by hierarchy if specified
            if query.hierarchy_ids:
                if query.hierarchy_level == HierarchyLevel.STORE:
                    df = df[df["store_id"].isin(query.hierarchy_ids)]
                elif query.hierarchy_level == HierarchyLevel.CATEGORY:
                    df = df[df["category_id"].isin(query.hierarchy_ids)]

            # Convert to MetricData objects
            for _, row in df.iterrows():
                if query.hierarchy_level == HierarchyLevel.STORE:
                    hierarchy_id = row["store_id"]
                    hierarchy_name = row["store_name"]
                elif query.hierarchy_level == HierarchyLevel.CATEGORY:
                    hierarchy_id = row["category_id"]
                    hierarchy_name = row["category_name"]
                else:
                    hierarchy_id = row["store_id"]
                    hierarchy_name = row["store_name"]

                metric_data = MetricData(
                    metric_name=metric,
                    value=row["value"],
                    week_ending=row["week_ending"].date(),
                    hierarchy_level=query.hierarchy_level,
                    hierarchy_id=hierarchy_id,
                    hierarchy_name=hierarchy_name,
                    budget=row["budget"] if query.include_budget else None
                )
                all_data.append(metric_data)

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            data=all_data,
            row_count=len(all_data),
            execution_time_ms=execution_time
        )

    def get_weekly_comparisons(
        self,
        metric: MetricType,
        hierarchy_level: HierarchyLevel,
        current_week: Optional[date] = None
    ) -> List[WeeklyComparison]:
        """
        Get week-over-week comparison data.

        Args:
            metric: The metric to compare
            hierarchy_level: Level of aggregation
            current_week: Current week ending date (defaults to most recent)

        Returns:
            List of WeeklyComparison objects
        """
        df = self._load_metric_data(metric)

        # Determine current week if not specified
        if current_week is None:
            current_week = df["week_ending"].max().date()

        previous_week = current_week - timedelta(weeks=1)

        # Aggregate by hierarchy level
        if hierarchy_level == HierarchyLevel.STORE:
            group_cols = ["week_ending", "store_id", "store_name"]
            id_col = "store_id"
            name_col = "store_name"
        elif hierarchy_level == HierarchyLevel.CATEGORY:
            group_cols = ["week_ending", "category_id", "category_name"]
            id_col = "category_id"
            name_col = "category_name"
        elif hierarchy_level == HierarchyLevel.NATIONAL:
            # Aggregate everything
            df = df.groupby("week_ending").agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            df["hierarchy_id"] = "NAT001"
            df["hierarchy_name"] = "Woolworths Australia"
            group_cols = ["week_ending", "hierarchy_id", "hierarchy_name"]
            id_col = "hierarchy_id"
            name_col = "hierarchy_name"
        else:
            # Default to store level
            group_cols = ["week_ending", "store_id", "store_name"]
            id_col = "store_id"
            name_col = "store_name"

        if hierarchy_level != HierarchyLevel.NATIONAL:
            agg_df = df.groupby(group_cols).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df[id_col]
            agg_df["hierarchy_name"] = agg_df[name_col]
        else:
            agg_df = df

        # Get current and previous week data
        current_data = agg_df[agg_df["week_ending"] == pd.Timestamp(current_week)]
        previous_data = agg_df[agg_df["week_ending"] == pd.Timestamp(previous_week)]

        # Merge
        merged = current_data.merge(
            previous_data[["hierarchy_id", "value"]],
            on="hierarchy_id",
            suffixes=("", "_prev"),
            how="inner"
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

    def get_aggregated_data(
        self,
        metric: MetricType,
        hierarchy_level: HierarchyLevel,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get aggregated data for a metric at a hierarchy level.

        Args:
            metric: The metric to retrieve
            hierarchy_level: Level of aggregation
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Aggregated DataFrame
        """
        df = self._load_metric_data(metric)

        # Filter by date
        df = df[
            (df["week_ending"] >= pd.Timestamp(start_date)) &
            (df["week_ending"] <= pd.Timestamp(end_date))
        ]

        # Aggregate based on level
        if hierarchy_level == HierarchyLevel.STORE:
            agg_df = df.groupby(["week_ending", "store_id", "store_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["store_id"]
            agg_df["hierarchy_name"] = agg_df["store_name"]

        elif hierarchy_level == HierarchyLevel.CATEGORY:
            agg_df = df.groupby(["week_ending", "category_id", "category_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["category_id"]
            agg_df["hierarchy_name"] = agg_df["category_name"]

        elif hierarchy_level == HierarchyLevel.NATIONAL:
            agg_df = df.groupby("week_ending").agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = "NAT001"
            agg_df["hierarchy_name"] = "Woolworths Australia"

        else:
            # Default aggregation by store
            agg_df = df.groupby(["week_ending", "store_id", "store_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["store_id"]
            agg_df["hierarchy_name"] = agg_df["store_name"]

        agg_df["metric"] = metric.value
        agg_df["hierarchy_level"] = hierarchy_level.value

        return agg_df

    def get_hierarchy_data(
        self,
        metric: MetricType,
        level: HierarchyLevel,
        current_week: date
    ) -> List[Dict]:
        """
        Get data for a specific hierarchy level formatted for insights.

        Args:
            metric: The metric to analyze
            level: Hierarchy level
            current_week: Current week ending date

        Returns:
            List of dictionaries with hierarchy data
        """
        comparisons = self.get_weekly_comparisons(metric, level, current_week)

        return [
            {
                "id": c.hierarchy_id,
                "name": c.hierarchy_name,
                "current_value": c.current_value,
                "previous_value": c.previous_week_value,
                "wow_change_pct": c.wow_change_pct,
                "budget_variance_pct": c.budget_variance_pct
            }
            for c in comparisons
        ]

    def generate_sql(self, query: DataQuery) -> str:
        """
        Generate a BigQuery SQL query (for documentation/reference).

        Args:
            query: The query specification

        Returns:
            SQL query string
        """
        metrics_str = ", ".join([f"'{m.value}'" for m in query.metrics])
        hierarchy_filter = ""

        if query.hierarchy_ids:
            ids_str = ", ".join([f"'{id}'" for id in query.hierarchy_ids])
            hierarchy_filter = f"AND hierarchy_id IN ({ids_str})"

        sql = f"""
SELECT
    week_ending,
    metric_name,
    hierarchy_level,
    hierarchy_id,
    hierarchy_name,
    value,
    budget,
    LAG(value) OVER (PARTITION BY hierarchy_id, metric_name ORDER BY week_ending) as previous_value
FROM
    `woolworths.metrics.weekly_aggregates`
WHERE
    metric_name IN ({metrics_str})
    AND hierarchy_level = '{query.hierarchy_level.value}'
    AND week_ending BETWEEN '{query.start_date}' AND '{query.end_date}'
    {hierarchy_filter}
ORDER BY
    week_ending DESC, hierarchy_id
"""
        return sql.strip()

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
