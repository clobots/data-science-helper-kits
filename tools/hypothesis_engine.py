"""Statistical hypothesis detection engine for identifying significant metric changes."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class HypothesisEngine:
    """
    Statistical engine for detecting significant changes in metrics.

    Simple tests:
    - percentile_outlier: Is value in top 5% or bottom 5%?
    - above_average_change: Is change > average change?
    - budget_variance: Is variance from budget > threshold?
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "hypothesis_config.json"

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the hypothesis engine.

        Args:
            config_path: Path to JSON config file (uses default if not provided)
        """
        config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from JSON file."""
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {
            "tests": {
                "percentile_outlier": {"enabled": True, "upper_percentile": 95, "lower_percentile": 5},
                "above_average_change": {"enabled": True},
                "budget_variance": {"enabled": True, "threshold_pct": 3.0}
            },
            "defaults": {"min_data_points": 3, "flag_insight_ready_when_any_sig": True}
        }

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

        Returns DataFrame with columns:
        - category, metric, test_type, test_metric, sig_flag, direction, flag_insight_ready
        """
        results = []
        tests_config = self.config.get("tests", {})

        # Group by category and metric
        for (category, metric), group_df in df.groupby([category_col, metric_col]):
            group_df = group_df.sort_values(week_col)
            latest = group_df.iloc[-1]

            # Test 1: Percentile outlier
            if tests_config.get("percentile_outlier", {}).get("enabled", True):
                result = self._test_percentile_outlier(
                    group_df, value_col, category, metric
                )
                if result:
                    results.append(result)

            # Test 2: Above average change
            if tests_config.get("above_average_change", {}).get("enabled", True):
                result = self._test_above_average_change(
                    group_df, value_col, prev_value_col, category, metric
                )
                if result:
                    results.append(result)

            # Test 3: Budget variance
            if tests_config.get("budget_variance", {}).get("enabled", True):
                result = self._test_budget_variance(
                    latest, value_col, budget_col, category, metric
                )
                if result:
                    results.append(result)

        # Create output DataFrame
        if results:
            output_df = pd.DataFrame(results)
            output_df = output_df.sort_values(
                ["sig_flag", "test_metric"],
                ascending=[False, False],
                key=lambda x: x.abs() if x.name == "test_metric" else x
            )
        else:
            output_df = pd.DataFrame(columns=[
                "category", "metric", "test_type", "test_metric",
                "sig_flag", "direction", "flag_insight_ready"
            ])

        return output_df.reset_index(drop=True)

    def _test_percentile_outlier(
        self,
        df: pd.DataFrame,
        value_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """Test if latest value is an outlier (outside 5th-95th percentile)."""
        config = self.config.get("tests", {}).get("percentile_outlier", {})
        upper_pct = config.get("upper_percentile", 95)
        lower_pct = config.get("lower_percentile", 5)

        values = df[value_col].values
        if len(values) < self.config.get("defaults", {}).get("min_data_points", 3):
            return None

        latest_value = values[-1]
        p_upper = np.percentile(values, upper_pct)
        p_lower = np.percentile(values, lower_pct)

        is_outlier = latest_value > p_upper or latest_value < p_lower

        if latest_value > p_upper:
            direction = "up"
            # How far above the 95th percentile (as %)
            test_metric = ((latest_value - p_upper) / p_upper * 100) if p_upper != 0 else 0
        elif latest_value < p_lower:
            direction = "down"
            # How far below the 5th percentile (as %)
            test_metric = ((p_lower - latest_value) / p_lower * 100) if p_lower != 0 else 0
        else:
            direction = "stable"
            test_metric = 0

        return {
            "category": category,
            "metric": metric,
            "test_type": "percentile_outlier",
            "test_metric": round(test_metric, 2),
            "sig_flag": is_outlier,
            "direction": direction,
            "flag_insight_ready": is_outlier
        }

    def _test_above_average_change(
        self,
        df: pd.DataFrame,
        value_col: str,
        prev_value_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """Test if latest change exceeds average historical change."""
        latest = df.iloc[-1]
        current_value = latest[value_col]
        prev_value = latest[prev_value_col]

        if pd.isna(prev_value) or prev_value == 0:
            return None

        # Calculate current change %
        current_change = ((current_value - prev_value) / abs(prev_value)) * 100

        # Calculate historical average absolute change
        values = df[value_col].values
        if len(values) < 2:
            return None

        changes = []
        for i in range(1, len(values)):
            if values[i-1] != 0:
                pct_change = abs((values[i] - values[i-1]) / values[i-1]) * 100
                changes.append(pct_change)

        if not changes:
            return None

        avg_change = np.mean(changes)
        is_above_avg = abs(current_change) > avg_change

        direction = "up" if current_change > 0 else "down" if current_change < 0 else "stable"

        return {
            "category": category,
            "metric": metric,
            "test_type": "above_avg_change",
            "test_metric": round(current_change, 2),
            "sig_flag": is_above_avg,
            "direction": direction,
            "flag_insight_ready": is_above_avg
        }

    def _test_budget_variance(
        self,
        latest_row: pd.Series,
        value_col: str,
        budget_col: str,
        category: str,
        metric: str
    ) -> Optional[dict]:
        """Test if variance from budget exceeds threshold."""
        config = self.config.get("tests", {}).get("budget_variance", {})
        threshold = config.get("threshold_pct", 3.0)

        current_value = latest_row[value_col]
        budget_value = latest_row[budget_col]

        if pd.isna(budget_value) or budget_value == 0:
            return None

        variance_pct = ((current_value - budget_value) / abs(budget_value)) * 100
        is_significant = abs(variance_pct) > threshold

        direction = "up" if variance_pct > 0 else "down" if variance_pct < 0 else "stable"

        return {
            "category": category,
            "metric": metric,
            "test_type": "budget_variance",
            "test_metric": round(variance_pct, 2),
            "sig_flag": is_significant,
            "direction": direction,
            "flag_insight_ready": is_significant
        }

    def analyze_from_csv(self, csv_path: Path, **kwargs) -> pd.DataFrame:
        """Analyze metrics from a CSV file."""
        df = pd.read_csv(csv_path, parse_dates=["week_ending"])
        return self.analyze(df, **kwargs)

    def get_flagged(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows where flag_insight_ready is True."""
        return results_df[results_df["flag_insight_ready"] == True].copy()

    def summary(self, results_df: pd.DataFrame) -> dict:
        """Generate summary statistics."""
        total = len(results_df)
        sig_count = results_df["sig_flag"].sum()

        return {
            "total_tests": total,
            "significant_count": int(sig_count),
            "by_test_type": results_df.groupby("test_type")["sig_flag"].sum().to_dict(),
            "by_direction": results_df[results_df["sig_flag"]].groupby("direction").size().to_dict()
        }


def run_hypothesis_analysis(
    csv_path: Optional[Path] = None,
    df: Optional[pd.DataFrame] = None,
    config_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, dict]:
    """Run hypothesis analysis and return results with summary."""
    engine = HypothesisEngine(config_path=config_path)

    if df is not None:
        results = engine.analyze(df)
    elif csv_path:
        results = engine.analyze_from_csv(csv_path)
    else:
        default_path = Path(__file__).parent.parent / "data" / "sample_data" / "sample_metrics.csv"
        results = engine.analyze_from_csv(default_path)

    return results, engine.summary(results)


if __name__ == "__main__":
    results, summary = run_hypothesis_analysis()

    print("=" * 70)
    print("HYPOTHESIS ENGINE RESULTS")
    print("=" * 70)
    print(f"\n{results.to_string()}")
    print("\n" + "-" * 70)
    print("SUMMARY:", summary)
    print("\n" + "-" * 70)
    print("FLAGGED FOR INSIGHTS:")
    flagged = results[results["flag_insight_ready"] == True]
    print(flagged.to_string() if len(flagged) > 0 else "  None")
