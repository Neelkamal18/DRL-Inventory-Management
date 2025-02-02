import logging
import json
import pandas as pd
import random
import numpy as np
from pathlib import Path
import typing

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration for generating initial condition (IC) data.

    Attributes:
        raw_data_path (Path): Folder where raw data is located.
        instance_path (Path): Folder where the generated files will be stored.
        instance_name (str): Name prefix for output files.
        items_file (str): Name of the product master list file.
        forecast_file (str): Name of the forecast data file.
        open_positions_file (str): Name of the open purchase orders file.
        num_instances (int): Number of IC instances to generate.
        num_products (int): Number of products to include (set to None for all).
        random_seed (int): Random seed for reproducibility.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent / "data"
    raw_data_path: Path = BASE_DIR / "raw"
    instance_path: Path = Path(__file__).resolve().parent.parent / "SAA"
    
    instance_name: str = "ic_instance"
    items_file: str = "item_master_50.csv"
    forecast_file: str = "forecast.csv"
    open_positions_file: str = "open_po.csv"

    num_instances: int = 200  # Number of instances to generate
    num_products: int = 15  # Number of products to keep (None for all)
    random_seed: int = 42  # Set seed for reproducibility


class Product(typing.NamedTuple):
    """Represents a single product with an ID and on-hand inventory."""

    id: int
    on_hand_inventory: int

    def to_json(self) -> dict:
        """Converts the product instance into a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "on_hand_inventory": self.on_hand_inventory
        }


class IcData(typing.NamedTuple):
    """Holds initial condition (IC) data including configuration and product details."""

    config: Config
    products: list[Product]

    @classmethod
    def build(cls, cfg: Config) -> "IcData":
        """Constructs the IC dataset from raw data.

        Steps:
        1. Reads and filters the master product list.
        2. Selects a subset of products based on configurable parameters.
        3. Generates randomized inventory values for products.
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

        # Generate randomized inventory values
        products = [Product(id=int(row["item"]), on_hand_inventory=random.randint(0, 5)) for _, row in df.iterrows()]

        return cls(config=cfg, products=products)

    def write_to_file(self, fpath: Path):
        """Writes the IC data to a JSON file."""
        
        cfg_dict = self.config._asdict()
        excluded_keys = {"raw_data_path", "instance_path", "items_file", "forecast_file", "open_positions_file", "num_instances", "num_products", "random_seed"}
        
        # Remove unnecessary fields before writing to file
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}

        data = {
            "config": filtered_cfg,
            "products": [p.to_json() for p in self.products]
        }

        with open(fpath, "w") as outfile:
            json.dump(data, outfile, indent=4)

        log.info(f"Successfully wrote IC instance to {fpath}")


def main():
    """Generates multiple IC data instances and writes them to files."""
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --: %(message)s",
        level=logging.DEBUG
    )

    cfg = Config()
    log.info("Starting IC data creation...")

    # Ensure the output directory exists
    cfg.instance_path.mkdir(parents=True, exist_ok=True)

    for i in range(cfg.num_instances):
        ic_data = IcData.build(cfg)
        output_file = cfg.instance_path / f"{cfg.instance_name}_{i}.json"
        ic_data.write_to_file(output_file)

    log.info(f"Successfully wrote {cfg.num_instances} IC instances to {cfg.instance_path}")


if __name__ == "__main__":
    main()
