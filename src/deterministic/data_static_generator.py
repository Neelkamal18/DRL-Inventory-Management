import logging
import json
import pandas as pd
import random
import numpy as np
import scipy.stats as stats
from pathlib import Path
import typing

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration for generating static demand scenarios.

    Attributes:
        raw_data_path (Path): Folder where raw data is located.
        instance_path (Path): Folder where the generated file will be stored.
        instance_name (str): Name of the output file.
        num_time_points (int): Number of time points for demand forecasting.
        location_id (int): Location ID to filter raw data.
        lead_time_in_weeks (int): Lead time in weeks.
        container_volume (float): Volume of each container.
        items_file (str): Name of the product master list file.
        forecast_file (str): Name of the forecast data file.
        open_positions_file (str): Name of the open purchase orders file.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    raw_data_path: Path = BASE_DIR / "raw"
    instance_path: Path = BASE_DIR
    instance_name: str = "static_instance"
    
    num_time_points: int = 3
    location_id: int = 3000
    lead_time_in_weeks: int = 12
    container_volume: float = 2350.0

    items_file: str = "item_master_50.csv"
    forecast_file: str = "forecast.csv"
    open_positions_file: str = "open_po.csv"

    num_products: int = 15  # Number of products to include (configurable)
    random_seed: int = 42  # Set seed for reproducibility


class Product(typing.NamedTuple):
    """Represents a product with attributes for MILP modeling."""
    
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
    """Holds static instance data including demand forecasts and cost factors."""

    config: Config
    products: list[Product]
    product_mle_params: dict[int, dict[str, float]]
    demand_data: dict[str, int]
    ramping_factor: dict[int, int]
    F: dict[str, float]  # Cost of failing to meet demand
    H: dict[str, float]  # Cost of overstocking
    G: dict[str, float]  # Reward for maintaining recommended stock
    alpha: float  # Proportionality cost penalty

    @classmethod
    def build(cls, cfg: Config) -> "StaticData":
        """Builds a StaticData instance from raw product and forecast data."""

        # Load product master list
        items_file = cfg.raw_data_path / cfg.items_file
        df = pd.read_csv(items_file)

        # Remove items with zero safety stock and zero on-hand inventory
        df = df.query("on_hand != 0 or safety_stock != 0")

        # Ensure reproducibility
        random.seed(cfg.random_seed)

        # Select subset of products (configurable)
        df = df.head(cfg.num_products)

        # Create product objects
        products = [
            Product(id=int(row["item"]), volume=row["volume"], safety_stock_in_weeks=row["safety_stock"])
            for _, row in df.iterrows()
        ]
        product_ids = {p.id for p in products}

        # ---- Load Demand Forecast and Fit Gamma Distribution ----
        forecast_file = cfg.raw_data_path / cfg.forecast_file
        df_forecast = pd.read_csv(forecast_file)

        # Filter forecast data for selected product IDs
        df_forecast = df_forecast[df_forecast["item"].isin(product_ids)]
        df_forecast["units"] = df_forecast["units"].apply(np.ceil).astype(int).abs()

        mle_params = {}
        demand_data = {}

        for product_id in product_ids:
            product_forecast = df_forecast[df_forecast["item"] == product_id]["units"].values

            if len(product_forecast) > 0:
                # Fit Gamma distribution
                alpha, loc, scale = stats.gamma.fit(product_forecast)
                mle_params[product_id] = {"alpha": alpha, "loc": loc, "scale": scale}
            else:
                # Default parameters in case data is missing
                mle_params[product_id] = {"alpha": 2.0, "loc": 0.0, "scale": 1.0}

            # Generate demand values
            demand_values = [random.randint(5, 15) for _ in range(cfg.num_time_points)]
            #demand_values = stats.gamma.rvs(alpha, loc=loc, scale=scale, size=cfg.num_time_points)
            #demand_values = np.ceil(demand).astype(int)
            
            for week in range(cfg.num_time_points):
                demand_data[f"{product_id}_{week}"] = demand_values[week]

        # ---- Generate Cost Parameters ----
        F = {str(pid): random.uniform(30, 40) for pid in product_ids}  # Cost of failing to meet demand
        H = {str(pid): F[str(pid)] / 2 for pid in product_ids}  # Overstocking penalty
        G = {str(pid): F[str(pid)] / 3 for pid in product_ids}  # Reward for maintaining safety stock

        # ---- Define Ramping Factor ----
        ramping_factor = {str(pid): random.randint(1, 5) for pid in product_ids}

        return cls(
            config=cfg,
            products=products,
            product_mle_params=mle_params,
            demand_data=demand_data,
            ramping_factor=ramping_factor,
            F=F,
            H=H,
            G=G,
            alpha=25.0
        )

    def write_to_file(self, fpath: Path):
        """Writes the StaticData instance to a JSON file."""
        
        cfg_dict = self.config._asdict()
        excluded_keys = {"raw_data_path", "instance_path", "items_file", "forecast_file", "open_positions_file", "num_products", "random_seed"}
        
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

        log.info(f"Successfully wrote StaticData instance to {fpath}")


def main():
    """Generates static demand scenarios and writes them to a file."""
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --: %(message)s",
        level=logging.DEBUG
    )

    cfg = Config()
    log.info("Starting static data creation...")

    static_data = StaticData.build(cfg)
    output_file = cfg.instance_path / f"{cfg.instance_name}.json"

    static_data.write_to_file(output_file)
    log.info(f"Instance successfully written to {output_file}")


if __name__ == "__main__":
    main()
