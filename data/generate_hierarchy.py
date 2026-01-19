"""Generate realistic Australian Woolworths store hierarchy and metrics data."""

import random
import string
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Australian hierarchy structure
# State -> Zones -> Groups -> Stores (5 per group)
HIERARCHY = {
    "NSW": {
        "zones": {
            "NSW_Metro": ["Sydney_CBD", "Sydney_North", "Sydney_South", "Sydney_West"],
            "NSW_Hunter": ["Newcastle", "Central_Coast"],
            "NSW_Regional": ["Wollongong", "Canberra_Border"],
            "NSW_North": ["North_Coast", "New_England"]
        }
    },
    "VIC": {
        "zones": {
            "VIC_Metro": ["Melbourne_CBD", "Melbourne_East", "Melbourne_West", "Melbourne_South"],
            "VIC_Regional": ["Geelong", "Ballarat", "Bendigo"],
            "VIC_Gippsland": ["Gippsland_East", "Gippsland_West"]
        }
    },
    "QLD": {
        "zones": {
            "QLD_SEQ": ["Brisbane_CBD", "Brisbane_North", "Brisbane_South", "Gold_Coast"],
            "QLD_Central": ["Sunshine_Coast", "Toowoomba"],
            "QLD_North": ["Townsville", "Cairns"]
        }
    },
    "WA": {
        "zones": {
            "WA_Metro": ["Perth_CBD", "Perth_North", "Perth_South"],
            "WA_Regional": ["Bunbury", "Geraldton"]
        }
    },
    "SA": {
        "zones": {
            "SA_Metro": ["Adelaide_CBD", "Adelaide_North", "Adelaide_South"],
            "SA_Regional": ["Mount_Gambier"]
        }
    },
    "TAS": {
        "zones": {
            "TAS_South": ["Hobart", "Kingston"],
            "TAS_North": ["Launceston"]
        }
    },
    "NT": {
        "zones": {
            "NT_Top_End": ["Darwin", "Palmerston"]
        }
    },
    "ACT": {
        "zones": {
            "ACT_Central": ["Canberra_North", "Canberra_South"]
        }
    }
}

STORES_PER_GROUP = 5
WEEKS = 5
METRICS = ["wages", "voice_of_customer", "sales", "stockloss", "order_pickrate"]

# Metric configs: (base_mean, std_dev, budget_mean, lower_is_better)
METRIC_CONFIG = {
    "wages": (12.5, 1.5, 12.0, True),
    "voice_of_customer": (82, 5, 82, False),
    "sales": (450000, 80000, 460000, False),
    "stockloss": (1.2, 0.3, 1.2, True),
    "order_pickrate": (95, 6, 95, False)
}


def generate_store_id() -> str:
    """Generate unique 6-char alphanumeric store ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


def generate_hierarchy_data():
    """Generate the complete hierarchy structure."""
    records = []
    store_ids = set()

    for state, state_data in HIERARCHY.items():
        for zone, groups in state_data["zones"].items():
            for group in groups:
                for i in range(STORES_PER_GROUP):
                    # Generate unique store ID
                    while True:
                        store_id = generate_store_id()
                        if store_id not in store_ids:
                            store_ids.add(store_id)
                            break

                    store_name = f"{group.replace('_', ' ')} Store {i+1}"

                    records.append({
                        "store_id": store_id,
                        "store_name": store_name,
                        "group": group,
                        "zone": zone,
                        "state": state,
                        "national": "Woolworths_AU"
                    })

    return pd.DataFrame(records)


def generate_metrics_data(hierarchy_df: pd.DataFrame):
    """Generate metrics data for all stores across weeks."""
    records = []

    # Generate week ending dates (Sundays)
    end_date = date.today()
    days_to_sunday = (6 - end_date.weekday()) % 7
    end_date = end_date + timedelta(days=days_to_sunday)

    week_dates = [end_date - timedelta(weeks=i) for i in range(WEEKS)]
    week_dates.reverse()

    # Create some "problem" stores and "star" stores for interesting data
    all_stores = hierarchy_df["store_id"].tolist()
    problem_stores = set(random.sample(all_stores, max(5, len(all_stores) // 20)))
    star_stores = set(random.sample([s for s in all_stores if s not in problem_stores],
                                     max(5, len(all_stores) // 20)))

    for _, store_row in hierarchy_df.iterrows():
        store_id = store_row["store_id"]

        # Store-level modifier for consistency
        store_modifier = np.random.uniform(0.9, 1.1)

        # Determine store type
        is_problem = store_id in problem_stores
        is_star = store_id in star_stores

        for metric in METRICS:
            base_mean, std_dev, budget_mean, lower_is_better = METRIC_CONFIG[metric]

            prev_value = None

            for week_idx, week_date in enumerate(week_dates):
                # Base value
                value = np.random.normal(base_mean * store_modifier, std_dev)

                # Apply trend for problem/star stores
                if is_problem:
                    if lower_is_better:
                        value *= (1 + 0.05 * week_idx)  # Getting worse
                    else:
                        value *= (1 - 0.03 * week_idx)  # Getting worse
                elif is_star:
                    if lower_is_better:
                        value *= (1 - 0.03 * week_idx)  # Getting better
                    else:
                        value *= (1 + 0.02 * week_idx)  # Getting better

                # Ensure reasonable bounds
                if metric == "voice_of_customer":
                    value = np.clip(value, 50, 100)
                elif metric == "wages":
                    value = np.clip(value, 8, 20)
                elif metric == "stockloss":
                    value = np.clip(value, 0.3, 4.0)
                elif metric == "order_pickrate":
                    value = np.clip(value, 60, 120)
                elif metric == "sales":
                    value = max(value, 100000)

                value = round(value, 2)
                budget = round(budget_mean * store_modifier, 2)

                records.append({
                    "week_ending": week_date,
                    "store_id": store_id,
                    "store_name": store_row["store_name"],
                    "group": store_row["group"],
                    "zone": store_row["zone"],
                    "state": store_row["state"],
                    "metric": metric,
                    "value": value,
                    "budget": budget,
                    "previous_week_value": prev_value if prev_value else value * np.random.uniform(0.97, 1.03)
                })

                prev_value = value

    return pd.DataFrame(records)


def main():
    output_dir = Path(__file__).parent / "sample_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating hierarchy...")
    hierarchy_df = generate_hierarchy_data()
    hierarchy_path = output_dir / "hierarchy.csv"
    hierarchy_df.to_csv(hierarchy_path, index=False)
    print(f"  Created {len(hierarchy_df)} stores")
    print(f"  Saved to {hierarchy_path}")

    print("\nHierarchy summary:")
    print(f"  States: {hierarchy_df['state'].nunique()}")
    print(f"  Zones: {hierarchy_df['zone'].nunique()}")
    print(f"  Groups: {hierarchy_df['group'].nunique()}")
    print(f"  Stores: {hierarchy_df['store_id'].nunique()}")

    print("\nGenerating metrics data...")
    metrics_df = generate_metrics_data(hierarchy_df)
    metrics_path = output_dir / "sample_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Created {len(metrics_df)} metric records")
    print(f"  Saved to {metrics_path}")

    print("\nSample hierarchy:")
    print(hierarchy_df.head(10).to_string())

    print("\nSample metrics:")
    print(metrics_df.head(10).to_string())


if __name__ == "__main__":
    main()
