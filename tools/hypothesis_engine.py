"""Statistical hypothesis detection engine for identifying significant metric changes."""

from datetime import date, timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import settings


class HypothesisEngine:
    """
    Statistical engine for detecting significant changes in metrics.

    Uses mathematical tests (z-scores, t-tests) rather than LLM inference.
    Outputs a structured DataFrame with significance flags.

    Test types:
    - wow: Week-over-week percentage change with z-score
    - budget: Variance from budget as percentage
    - trend: Linear regression slope significance (t-test)
    """

    # Significance level for statistical tests
    ALPHA = 0.05

    # Minimum data points for trend analysis
    MIN_TREND_POINTS = 4

    def __init__(
        self,
        wow_threshold: float = 5.0,
        budget_threshold: float = 3.0,
        trend_threshold: float = 0.05
    ):
        """
        Initialize the hypothesis engine.

        Args:
            wow_threshold: Minimum % change for WoW significance
            budget_threshold: Minimum % variance from budget for significance
            trend_threshold: P-value threshold for trend significance
        """
        self.wow_threshold = wow_threshold
        self.budget_threshold = budget_threshold
        self.trend_threshold = trend_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        category_col: str = "store_id",
        metric_col: str = "metric",
        value_col: str = "value",
        budget_col: str = "budget",
        prev_value_col: str = "previous_week_value",
        week_col: str = "week_ending"
    ) -> pd.DataFrame:
        """
        Analyze metrics data and return hypothesis table.

        Args:
            df: Input DataFrame with metric data
            category_col: Column name for category/store identifier
            metric_col: Column name for metric type
            value_col: Column name for current value
            budget_col: Column name for budget value
            prev_value_col: Column name for previous week value
            week_col: Column name for week ending date

        Returns:
            DataFrame with columns:
            - category: Store/category identifier
            - metric: Metric name
            - test_type: Type of test (wow, budget, trend)
            - test_metric: Statistical test value (z-score, t-stat, or % change)
            - p_value: P-value where applicable
            - sig_flag: Boolean significance flag
            - direction: 'up', 'down', or 'stable'
            - flag_insight_ready: Boolean indicating if ready for insights engine
        """
        results = []

        # Get unique category-metric combinations
        groups = df.groupby([category_col, metric_col])

        for (category, metric), group_df in groups:
            group_df = group_df.sort_values(week_col)

            # Test 1: Week-over-Week change
            wow_result = self._test_wow(
                group_df, value_col, prev_value_col, category, metric
            )
            if wow_result:
                results.append(wow_result)

            # Test 2: Budget variance
            budget_result = self._test_budget(
                group_df, value_col, budget_col, category, metric
            )
            if budget_result:
                results.append(budget_result)

            # Test 3: Trend analysis (if enough data points)
            if len(group_df) >= self.MIN_TREND_POINTS:
                trend_result = self._test_trend(
                    group_df, value_col, week_col, category, metric
                )
                if trend_result:
                    results.append(trend_result)

        # Create output DataFrame
        if results:
            output_df = pd.DataFrame(results)
            # Sort by significance and test metric magnitude
            output_df = output_df.sort_values(
                ["sig_flag", "test_metric"],
                ascending=[False, False],
                key=lambda x: x.abs() if x.name == "test_metric" else x
            )
        else:
            output_df = pd.DataFrame(columns=[
                "category", "metric", "test_type", "test_metric",
                "p_value", "sig_flag", "direction", "flag_insight_ready"
            ])

        return output_df.reset_index(drop=True)

    def _test_wow(
        self,
        df: pd.DataFrame,
        value_col: str,
        prev_value_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """
        Test week-over-week change significance.

        Uses z-score based on historical WoW changes.
        """
        # Get most recent row
        latest = df.iloc[-1]
        current_value = latest[value_col]
        prev_value = latest[prev_value_col]

        if pd.isna(prev_value) or prev_value == 0:
            return None

        # Calculate percentage change
        pct_change = ((current_value - prev_value) / abs(prev_value)) * 100

        # Calculate z-score using historical changes if available
        if len(df) > 1:
            # Calculate historical WoW changes
            df_sorted = df.sort_values("week_ending")
            values = df_sorted[value_col].values
            historical_changes = []
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    change = ((values[i] - values[i-1]) / abs(values[i-1])) * 100
                    historical_changes.append(change)

            if len(historical_changes) > 1:
                mean_change = np.mean(historical_changes)
                std_change = np.std(historical_changes, ddof=1)
                if std_change > 0:
                    z_score = (pct_change - mean_change) / std_change
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = pct_change / self.wow_threshold
                    p_value = None
            else:
                z_score = pct_change / self.wow_threshold
                p_value = None
        else:
            z_score = pct_change / self.wow_threshold
            p_value = None

        # Determine significance
        sig_flag = abs(pct_change) >= self.wow_threshold

        # Determine direction
        if pct_change > 0.5:
            direction = "up"
        elif pct_change < -0.5:
            direction = "down"
        else:
            direction = "stable"

        return {
            "category": category,
            "metric": metric,
            "test_type": "wow",
            "test_metric": round(pct_change, 2),
            "p_value": round(p_value, 4) if p_value else None,
            "sig_flag": sig_flag,
            "direction": direction,
            "flag_insight_ready": sig_flag
        }

    def _test_budget(
        self,
        df: pd.DataFrame,
        value_col: str,
        budget_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """Test budget variance significance."""
        latest = df.iloc[-1]
        current_value = latest[value_col]
        budget_value = latest[budget_col]

        if pd.isna(budget_value) or budget_value == 0:
            return None

        # Calculate variance percentage
        variance_pct = ((current_value - budget_value) / abs(budget_value)) * 100

        # Significance based on threshold
        sig_flag = abs(variance_pct) >= self.budget_threshold

        # Direction relative to budget
        if variance_pct > 0.5:
            direction = "up"  # Over budget (or above target)
        elif variance_pct < -0.5:
            direction = "down"  # Under budget (or below target)
        else:
            direction = "stable"

        return {
            "category": category,
            "metric": metric,
            "test_type": "budget",
            "test_metric": round(variance_pct, 2),
            "p_value": None,
            "sig_flag": sig_flag,
            "direction": direction,
            "flag_insight_ready": sig_flag
        }

    def _test_trend(
        self,
        df: pd.DataFrame,
        value_col: str,
        week_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """
        Test for significant trend using linear regression.

        Uses t-test on the slope coefficient.
        """
        df_sorted = df.sort_values(week_col)
        values = df_sorted[value_col].values
        n = len(values)

        if n < self.MIN_TREND_POINTS:
            return None

        # Create time index (0, 1, 2, ...)
        x = np.arange(n)
        y = values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate t-statistic for slope
        t_stat = slope / std_err if std_err > 0 else 0

        # Significance based on p-value
        sig_flag = p_value < self.trend_threshold

        # Direction based on slope
        if slope > 0:
            direction = "up"
        elif slope < 0:
            direction = "down"
        else:
            direction = "stable"

        # Normalize slope as percentage of mean for interpretability
        mean_value = np.mean(values)
        if mean_value != 0:
            slope_pct = (slope / mean_value) * 100
        else:
            slope_pct = 0

        return {
            "category": category,
            "metric": metric,
            "test_type": "trend",
            "test_metric": round(t_stat, 2),
            "p_value": round(p_value, 4),
            "sig_flag": sig_flag,
            "direction": direction,
            "flag_insight_ready": sig_flag and abs(slope_pct) > 1.0
        }

    def analyze_from_csv(
        self,
        csv_path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """
        Analyze metrics from a CSV file.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments passed to analyze()

        Returns:
            Hypothesis results DataFrame
        """
        df = pd.read_csv(csv_path, parse_dates=["week_ending"])
        return self.analyze(df, **kwargs)

    def get_flagged_hypotheses(
        self,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return only rows where flag_insight_ready is True."""
        return results_df[results_df["flag_insight_ready"] == True].copy()

    def summary(self, results_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics from hypothesis results.

        Args:
            results_df: Output from analyze()

        Returns:
            Summary dictionary
        """
        total = len(results_df)
        sig_count = results_df["sig_flag"].sum()
        insight_ready = results_df["flag_insight_ready"].sum()

        by_test_type = results_df.groupby("test_type")["sig_flag"].sum().to_dict()
        by_direction = results_df[results_df["sig_flag"]].groupby("direction").size().to_dict()

        return {
            "total_tests": total,
            "significant_count": int(sig_count),
            "insight_ready_count": int(insight_ready),
            "by_test_type": by_test_type,
            "by_direction": by_direction,
            "significance_rate": round(sig_count / total * 100, 1) if total > 0 else 0
        }

    def to_markdown(self, results_df: pd.DataFrame) -> str:
        """Convert results DataFrame to markdown table."""
        return results_df.to_markdown(index=False)


# Convenience function for quick analysis
def run_hypothesis_analysis(
    csv_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    wow_threshold: float = 5.0,
    budget_threshold: float = 3.0
) -> Tuple[pd.DataFrame, dict]:
    """
    Run hypothesis analysis and return results with summary.

    Args:
        csv_path: Path to CSV file (uses default sample if not provided)
        df: DataFrame to analyze (alternative to csv_path)
        wow_threshold: WoW change threshold
        budget_threshold: Budget variance threshold

    Returns:
        Tuple of (results DataFrame, summary dict)
    """
    engine = HypothesisEngine(
        wow_threshold=wow_threshold,
        budget_threshold=budget_threshold
    )

    if df is not None:
        results = engine.analyze(df)
    elif csv_path:
        results = engine.analyze_from_csv(csv_path)
    else:
        # Use default sample data
        default_path = Path(__file__).parent.parent / "data" / "sample_data" / "sample_metrics.csv"
        results = engine.analyze_from_csv(default_path)

    summary = engine.summary(results)

    return results, summary


if __name__ == "__main__":
    # Demo run
    results, summary = run_hypothesis_analysis()

    print("=" * 60)
    print("HYPOTHESIS ENGINE RESULTS")
    print("=" * 60)
    print(f"\n{results.to_string()}")
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "-" * 60)
    print("FLAGGED FOR INSIGHTS (flag_insight_ready=True)")
    print("-" * 60)
    flagged = results[results["flag_insight_ready"] == True]
    print(flagged.to_string() if len(flagged) > 0 else "  None")
