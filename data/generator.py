"""Simulated Woolworths data generator for demo purposes."""

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import settings
from models.data_models import HierarchyLevel, HierarchyNode, MetricType


class WoolworthsDataGenerator:
    """Generates realistic simulated Woolworths supermarket data."""

    # Australian zones and states
    ZONES = {
        "QLD_North": ["QLD"],
        "QLD_South": ["QLD"],
        "NSW_Metro": ["NSW", "ACT"],
        "NSW_Regional": ["NSW"],
        "VIC_Metro": ["VIC"],
        "VIC_Regional": ["VIC", "TAS"],
    }

    # Product categories typical for Woolworths
    CATEGORIES = [
        "Fresh Produce", "Meat & Seafood", "Dairy & Eggs", "Bakery",
        "Pantry", "Frozen", "Drinks", "Snacks & Confectionery",
        "Health & Beauty", "Baby", "Pet", "Household",
        "Liquor", "Deli", "Ready Meals", "International Foods",
        "Organic", "Gluten Free", "Plant Based", "Entertaining"
    ]

    # Metric baseline configurations (mean, std_dev, lower_is_better)
    METRIC_CONFIGS = {
        MetricType.WAGES: (12.5, 1.5, True),  # Percentage of sales
        MetricType.VOICE_OF_CUSTOMER: (82.0, 5.0, False),  # Score 0-100
        MetricType.SALES: (500000.0, 100000.0, False),  # Weekly sales $
        MetricType.STOCKLOSS: (1.2, 0.3, True),  # Percentage
        MetricType.ORDER_PICKRATE: (95.0, 8.0, False),  # Items per hour
    }

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.hierarchy = self._build_hierarchy()
        self._anomaly_stores: Dict[str, List[Tuple[MetricType, str]]] = {}

    def _build_hierarchy(self) -> Dict[str, List[HierarchyNode]]:
        """Build the organizational hierarchy."""
        hierarchy = {
            "national": [],
            "zone": [],
            "state": [],
            "group": [],
            "store": [],
            "category": []
        }

        # National level
        national = HierarchyNode(
            level=HierarchyLevel.NATIONAL,
            id="NAT001",
            name="Woolworths Australia"
        )
        hierarchy["national"].append(national)

        # Zones
        for zone_name, states in self.ZONES.items():
            zone = HierarchyNode(
                level=HierarchyLevel.ZONE,
                id=f"ZONE_{zone_name}",
                name=zone_name,
                parent_id="NAT001"
            )
            hierarchy["zone"].append(zone)

            # States within zone
            for state_name in states:
                state_id = f"STATE_{state_name}_{zone_name}"
                state = HierarchyNode(
                    level=HierarchyLevel.STATE,
                    id=state_id,
                    name=f"{state_name} ({zone_name})",
                    parent_id=zone.id
                )
                hierarchy["state"].append(state)

        # Generate groups and stores
        stores_per_group = settings.num_stores // 10
        group_counter = 0
        store_counter = 0

        for state in hierarchy["state"]:
            # 1-2 groups per state
            num_groups = random.randint(1, 2)
            for _ in range(num_groups):
                group_counter += 1
                group = HierarchyNode(
                    level=HierarchyLevel.GROUP,
                    id=f"GRP{group_counter:03d}",
                    name=f"Group {group_counter}",
                    parent_id=state.id
                )
                hierarchy["group"].append(group)

                # Stores per group
                for _ in range(stores_per_group):
                    store_counter += 1
                    store = HierarchyNode(
                        level=HierarchyLevel.STORE,
                        id=f"STR{store_counter:04d}",
                        name=f"Store {store_counter}",
                        parent_id=group.id
                    )
                    hierarchy["store"].append(store)

        # Categories (same for all stores)
        for cat_name in self.CATEGORIES[:settings.num_categories]:
            cat_id = cat_name.lower().replace(" ", "_").replace("&", "and")
            category = HierarchyNode(
                level=HierarchyLevel.CATEGORY,
                id=f"CAT_{cat_id}",
                name=cat_name
            )
            hierarchy["category"].append(category)

        return hierarchy

    def _inject_anomalies(self, stores: List[HierarchyNode]) -> None:
        """Select stores to have anomalies for testing hypothesis detection."""
        num_anomaly_stores = max(3, len(stores) // 10)
        anomaly_stores = random.sample(stores, num_anomaly_stores)

        for store in anomaly_stores:
            # Each anomaly store gets 1-2 metrics with issues
            num_issues = random.randint(1, 2)
            metrics = random.sample(list(MetricType), num_issues)

            for metric in metrics:
                # Type of anomaly: "spike", "drop", "trend_up", "trend_down"
                anomaly_type = random.choice(["spike", "drop", "trend_up", "trend_down"])

                if store.id not in self._anomaly_stores:
                    self._anomaly_stores[store.id] = []
                self._anomaly_stores[store.id].append((metric, anomaly_type))

    def _generate_metric_value(
        self,
        metric: MetricType,
        week_idx: int,
        store_id: str,
        base_modifier: float = 1.0
    ) -> Tuple[float, float]:
        """Generate a metric value with optional anomaly."""
        mean, std_dev, lower_is_better = self.METRIC_CONFIGS[metric]

        # Base value with some weekly variation
        weekly_seasonality = 1.0 + 0.05 * np.sin(2 * np.pi * week_idx / 12)
        value = np.random.normal(mean * weekly_seasonality * base_modifier, std_dev)

        # Check for anomalies
        if store_id in self._anomaly_stores:
            for anomaly_metric, anomaly_type in self._anomaly_stores[store_id]:
                if anomaly_metric == metric:
                    if anomaly_type == "spike" and week_idx >= settings.num_weeks - 2:
                        # Recent spike
                        multiplier = 1.3 if not lower_is_better else 1.4
                        value *= multiplier
                    elif anomaly_type == "drop" and week_idx >= settings.num_weeks - 2:
                        # Recent drop
                        multiplier = 0.7 if not lower_is_better else 0.6
                        value *= multiplier
                    elif anomaly_type == "trend_up":
                        # Gradual increase over time
                        trend_factor = 1.0 + (week_idx / settings.num_weeks) * 0.15
                        value *= trend_factor
                    elif anomaly_type == "trend_down":
                        # Gradual decrease over time
                        trend_factor = 1.0 - (week_idx / settings.num_weeks) * 0.15
                        value *= trend_factor

        # Budget is typically the target (slightly optimistic)
        budget = mean * base_modifier * 0.98 if lower_is_better else mean * base_modifier * 1.02

        # Ensure non-negative values
        value = max(0, value)

        return round(value, 2), round(budget, 2)

    def generate_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all simulated data."""
        # Inject anomalies
        self._inject_anomalies(self.hierarchy["store"])

        # Generate dates (weeks ending Sunday)
        end_date = date.today()
        # Adjust to most recent Sunday
        days_since_sunday = (end_date.weekday() + 1) % 7
        end_date = end_date - timedelta(days=days_since_sunday)

        dates = [end_date - timedelta(weeks=i) for i in range(settings.num_weeks)]
        dates.reverse()

        # Generate data for each metric
        data_frames = {}

        for metric in MetricType:
            records = []

            for store in self.hierarchy["store"]:
                # Store-level base modifier for variety
                store_modifier = np.random.uniform(0.85, 1.15)

                for cat in self.hierarchy["category"]:
                    for week_idx, week_date in enumerate(dates):
                        value, budget = self._generate_metric_value(
                            metric, week_idx, store.id, store_modifier
                        )

                        records.append({
                            "week_ending": week_date,
                            "store_id": store.id,
                            "store_name": store.name,
                            "category_id": cat.id,
                            "category_name": cat.name,
                            "value": value,
                            "budget": budget
                        })

            df = pd.DataFrame(records)
            data_frames[metric.value] = df

        return data_frames

    def save_data(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Generate and save data to CSV files."""
        output_dir = output_dir or settings.data_path
        output_dir.mkdir(parents=True, exist_ok=True)

        data_frames = self.generate_data()
        saved_files = {}

        for metric_name, df in data_frames.items():
            file_path = output_dir / f"{metric_name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[metric_name] = file_path
            print(f"Saved {len(df)} records to {file_path}")

        # Save hierarchy data
        hierarchy_records = []
        for level, nodes in self.hierarchy.items():
            for node in nodes:
                hierarchy_records.append({
                    "level": node.level,
                    "id": node.id,
                    "name": node.name,
                    "parent_id": node.parent_id
                })

        hierarchy_df = pd.DataFrame(hierarchy_records)
        hierarchy_path = output_dir / "hierarchy.csv"
        hierarchy_df.to_csv(hierarchy_path, index=False)
        saved_files["hierarchy"] = hierarchy_path
        print(f"Saved hierarchy with {len(hierarchy_df)} nodes to {hierarchy_path}")

        return saved_files

    def get_aggregated_data(
        self,
        metric: MetricType,
        level: HierarchyLevel,
        data_frames: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Aggregate data to a specified hierarchy level."""
        if data_frames is None:
            data_frames = self.generate_data()

        df = data_frames[metric.value].copy()

        if level == HierarchyLevel.CATEGORY:
            # Aggregate by category across all stores
            agg_df = df.groupby(["week_ending", "category_id", "category_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["category_id"]
            agg_df["hierarchy_name"] = agg_df["category_name"]

        elif level == HierarchyLevel.STORE:
            # Aggregate by store across all categories
            agg_df = df.groupby(["week_ending", "store_id", "store_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["store_id"]
            agg_df["hierarchy_name"] = agg_df["store_name"]

        elif level == HierarchyLevel.NATIONAL:
            # Aggregate all data
            agg_df = df.groupby("week_ending").agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = "NAT001"
            agg_df["hierarchy_name"] = "Woolworths Australia"

        else:
            # For group/state/zone, need to join with hierarchy
            # Simplified: just aggregate by store for demo
            agg_df = df.groupby(["week_ending", "store_id", "store_name"]).agg({
                "value": "mean",
                "budget": "mean"
            }).reset_index()
            agg_df["hierarchy_id"] = agg_df["store_id"]
            agg_df["hierarchy_name"] = agg_df["store_name"]

        agg_df["metric"] = metric.value
        agg_df["hierarchy_level"] = level.value

        return agg_df


if __name__ == "__main__":
    # Generate sample data when run directly
    generator = WoolworthsDataGenerator(seed=42)
    saved_files = generator.save_data()
    print(f"\nGenerated {len(saved_files)} data files")
