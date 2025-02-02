import logging
import json
import pandas as pd
import math
import csv
from pathlib import Path
import typing

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration class for MILP evaluation."""
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    optimal_action_file_path: Path = BASE_DIR / "milp_results" / "SAA_result_instance_action.csv"
    num_instances: int = 200
    saa_instance_path: Path = BASE_DIR / "SAA"
    saa_result_path: Path = BASE_DIR / "milp_results"


class Data(typing.NamedTuple):
    """Stores data parameters for MILP evaluation."""
    
    config: Config
    P: list[int]
    T: list[int]
    D: dict[str, int]
    L: int
    S: dict[int, float]
    V: dict[int, float]
    Vmax: float
    R: dict[str, int]
    F: dict[str, float]
    H: dict[str, float]
    G: dict[str, float]
    alpha: int
    init_inv: dict[int, int]
    N: int
    A: int
    B: int
    C: list[set[str]]

    @classmethod
    def build(cls, cfg: Config, ic_path: Path, static_path: Path) -> "Data":
        """Builds the data parameters from JSON files."""

        with open(static_path, "r") as file:
            static_data = json.load(file)

        products = static_data.get("products", [])
        config_data = static_data.get("config", {})

        product_ids = [p["id"] for p in products]
        week_numbers = list(range(config_data.get("num_time_points", 0)))
        demand = static_data.get("demand_data", {})
        lead_time = config_data["lead_time_in_weeks"]
        safety_stock = {p["id"]: math.floor(p["safety_stock_in_weeks"]) for p in products}
        volume = {p["id"]: p["volume"] for p in products}
        volume_max = config_data["container_volume"]
        ramping_units = static_data.get("ramping_factor", {})
        F = static_data.get("fail_demand_cost", {})
        H = static_data.get("overstocking_cost", {})
        G = static_data.get("reward_recommended_stock", {})
        alpha = static_data.get("proportionality_cost")

        with open(ic_path, "r") as file:
            ic_data = json.load(file)

        init_inv = {p["id"]: p["on_hand_inventory"] for p in ic_data.get("products", [])}

        # Construct parameter C (grouping products by safety stock values)
        distinct_safety_stock_values = set(safety_stock.values())
        C = [set(p_id for p_id, s_value in safety_stock.items() if s_value == s) for s in distinct_safety_stock_values]

        return cls(
            config=cfg,
            P=product_ids,
            T=week_numbers,
            D=demand,
            L=lead_time,
            S=safety_stock,
            V=volume,
            Vmax=volume_max,
            R=ramping_units,
            F=F,
            H=H,
            G=G,
            alpha=alpha,
            init_inv=init_inv,
            N=5,  # Number of containers
            A=10_000,  # Large constant 1
            B=10_000,  # Large constant 2
            C=C,
        )


class CalculateObjective:
    """Calculates the objective function value for MILP optimization."""

    def __init__(self, data: Data, cfg: Config, optimal_action: dict[str, int]):
        self.data = data
        self.optimal_action = optimal_action
        self.result_path = cfg.saa_result_path

        self.added_units = {f"{p}_{t}": 0 for p in data.P for t in data.T}
        self.unmet_units = {f"{p}_{t}": 0 for p in data.P for t in data.T}

        penalty_F = 0
        penalty_H = 0
        reward_G = 0

        for t in data.T:
            for p in data.P:
                key = f"{p}_{t}"
                prev_key = f"{p}_{t-1}" if t > 0 else None

                if t == 0:
                    total_inventory = data.init_inv[p] + optimal_action.get(key, 0)
                else:
                    total_inventory = self.added_units[prev_key] + optimal_action.get(key, 0)

                if total_inventory > data.D[key]:
                    self.added_units[key] = total_inventory - data.D[key]
                    self.unmet_units[key] = 0
                else:
                    self.added_units[key] = 0
                    self.unmet_units[key] = data.D[key] - total_inventory

        for t in data.T:
            for p in data.P:
                key = f"{p}_{t}"

                if self.added_units[key] == 0:
                    penalty_F += data.F[f"{p}"] * self.unmet_units[key]
                elif self.added_units[key] > 0:
                    overstock = self.added_units[key] - (data.D[key] * data.S[p])
                    if overstock > 0:
                        penalty_H += data.H[f"{p}"] * overstock
                    else:
                        reward_G -= data.G[f"{p}"] * self.added_units[key]

        self.reward = penalty_F + penalty_H + reward_G

    def write_to_csv(self):
        """Writes the calculated objective function value to a CSV file."""

        self.result_path.mkdir(parents=True, exist_ok=True)
        result_file = self.result_path / "SAA_result_instances.csv"

        file_exists = result_file.exists()

        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Instance Name", "Objective Function"])

            instance_count = sum(1 for _ in open(result_file, "r")) - 1  # Subtract header row
            writer.writerow([instance_count, self.reward])


def main():
    """Main function to evaluate MILP instances and write results."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)s --: %(message)s",
        level=logging.INFO,
    )

    cfg = Config()
    log.info("Loading optimal action data...")

    df = pd.read_csv(cfg.optimal_action_file_path)
    optimal_action = {
        f"{col}_{i}": row[col]
        for i, row in enumerate(df.to_dict(orient="records"))
        for col in df.columns[1:]
    }

    log.info("Starting MILP evaluation...")

    for i in range(cfg.num_instances):
        ic_filename = cfg.saa_instance_path / f"ic_instance_{i}.json"
        static_filename = cfg.saa_instance_path / f"static_instance_{i}.json"

        if not ic_filename.exists() or not static_filename.exists():
            log.warning(f"Skipping missing files: {ic_filename} or {static_filename}")
            continue

        data = Data.build(cfg, ic_filename, static_filename)
        log.info(f"Processing instance {i}...")

        result = CalculateObjective(data, cfg, optimal_action)
        result.write_to_csv()

    log.info("MILP evaluation completed.")


if __name__ == "__main__":
    main()
