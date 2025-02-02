import logging
import json
import pandas as pd
import numpy as np
import random
import math
import scipy.stats as stats
from pathlib import Path
import typing

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration for generating demand scenario instances.

    Attributes:
        raw_data_path (Path): Folder where raw data is located.
        instance_path (Path): Folder where the generated files will be stored.
        num_time_points (int): Number of time points for demand generation.
        instance_name (str): Name prefix for output files.
        lead_time_in_weeks (int): Lead time in weeks.
        container_volume (float): Volume of each container.
        items_file (str): Name of the product master list file.
        forecast_file (str): Name of the forecast data file.
        open_positions_file (str): Name of the open purchase orders file.
        num_instances (int): Number of static instances to generate.
        num_products (int): Number of products to include (set to None for all).
        random_seed (int): Random seed for reproducibility.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    raw_data_path: Path = BASE_DIR / "raw"
    instance_path: Path = Path(__file__).resolve().parent.parent / "SAA"
    
    instance_name: str = "static_instance"
    num_time_points: int = 3
    lead_time_in_weeks: int = 12
    container_volume: float = 2350.0
    items_file: str = "item_master.csv"
    forecast_file: str = "forecast.csv"
    open_positions_file: str = "open_po.csv"

    num_instances: int = 200  # Number of instances to generate
    num_products: int = 15  # Number of products to keep (None for all)
    random_seed: int = 42  # Set seed for reproducibility


class Product(typing.NamedTuple):
    """Represents a single product with volume and safety stock."""

    id: int
    volume: float
    safety_stock_in_weeks: int

    def to_json(self) -> dict:
        """Converts the product instance into a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "volume": self.volume,
            "safety_stock_in_weeks": self.safety_stock_in_weeks
        }


class StaticData(typing.NamedTuple):
    """Holds static instance data including configuration, product details, and demand scenarios."""

    config: Config
    products: list[Product]
    product_mle_params: dict[int, dict[str, float]]
    demand_data: dict[str, int]
    ramping_factor: dict[int, int]
    F: dict[str, float]
    H: dict[str, float]
    G: dict[str, float]
    alpha: int

    @classmethod
    def build(cls, cfg: Config) -> "StaticData":
        """Constructs the static dataset from raw data.

        Steps:
        1. Reads and filters the master product list.
        2. Generates demand scenarios and fits gamma distributions.
        3. Generates ramping factors and cost parameters.
        """

        # Load the master product file
        items_file = cfg.raw_data_path / cfg.items_file
        df = pd.read_csv(items_file)

        # Remove items with zero safety stock and zero on-hand inventory
        df = df.query("on_hand != 0 or safety_stock != 0")

        # Ensure reproducibility
        random.seed(cfg.random_seed)

        # Keep only a subset of products if specified
        if cfg.num_products:
            df = df.head(cfg.num_products)

        # Generate product objects
        products = [Product(id=int(row["item"]), volume=row["volume"], safety_stock_in_weeks=row["safety_stock"])
                    for _, row in df.iterrows()]

        product_ids = {p.id for p in products}

        # Read demand forecast
        forecast_file = cfg.raw_data_path / cfg.forecast_file
        df_forecast = pd.read_csv(forecast_file)

        # Filter forecast data for selected products
        df_forecast = df_forecast[df_forecast['item'].isin(product_ids)]
        df_forecast['units'] = df_forecast['units'].apply(math.ceil).abs()

        mle_params = {}
        demand_data = {}

        # Fit gamma distributions and generate demand scenarios
        for product_id in product_ids:
            df_item = df_forecast[df_forecast['item'] == product_id]

            if df_item.empty:
                continue

            observations = np.array(df_item['units'])
            alpha, loc, scale = stats.gamma.fit(observations)
            mle_params[product_id] = {"alpha": alpha, "loc": loc, "scale": scale}

            # Generate demand for each time point
            demand = np.ceil(stats.gamma.rvs(alpha, loc=loc, scale=scale, size=cfg.num_time_points)).astype(int)
            for week in range(cfg.num_time_points):
                key = f"{product_id}_{week}"
                demand_data[key] = abs(demand[week])

        # Generate ramping factors
        ramping_factor = {f"{id}": random.randint(5, 15) for id in product_ids}

        # Cost construction
        F = {f"{id}": random.uniform(30, 40) for id in product_ids}  # Penalty for unmet demand
        H = {f"{id}": F[f"{id}"] / 2 for id in product_ids}  # Overstocking penalty
        G = {f"{id}": F[f"{id}"] / 3 for id in product_ids}  # Reward for maintaining safety stock

        alpha = 25  # Penalty for violating proportionality

        return cls(
            config=cfg,
            products=products,
            product_mle_params=mle_params,
            demand_data=demand_data,
            ramping_factor=ramping_factor,
            F=F,
            H=H,
            G=G,
            alpha=alpha
        )

    def write_to_file(self, fpath: Path):
        """Writes the static data instance to a JSON file."""
        
        cfg_dict = self.config._asdict()
        excluded_keys = {"raw_data_path", "instance_path", "items_file", "forecast_file", "open_positions_file", "num_instances", "num_products", "random_seed"}
        
        # Remove unnecessary fields before writing to file
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}

        data = {
            "config": filtered_cfg,
            "products": [p.to_json() for p in self.products],
            "gamma_params": self.product_mle_params,
            "ramping_factor": self.ramping_factor,
            "demand_data": self.demand_data,
            "fail_demand_cost": self.F,
            "overstocking_cost": self.H,
            "reward_recommended_stock": self.G,
            "proportionality_cost": self.alpha
        }

        with open(fpath, "w") as outfile:
            json.dump(data, outfile, indent=4)

        log.info(f"Successfully wrote static instance to {fpath}")


def main():
    """Generates multiple static data instances and writes them to files."""
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --: %(message)s",
        level=logging.DEBUG
    )

    cfg = Config()
    log.info("Starting static data creation...")

    # Ensure the output directory exists
    cfg.instance_path.mkdir(parents=True, exist_ok=True)

    for i in range(cfg.num_instances):
        static_data = StaticData.build(cfg)
        output_file = cfg.instance_path / f"{cfg.instance_name}_{i}.json"
        static_data.write_to_file(output_file)

    log.info(f"Successfully wrote {cfg.num_instances} static instances to {cfg.instance_path}")


if __name__ == "__main__":
    main()
