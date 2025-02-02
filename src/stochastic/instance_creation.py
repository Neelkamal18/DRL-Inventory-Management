import logging
import json
import typing
import os
import numpy as np

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    """Configuration of SAA instance to be generated for MILP."""
    instance_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAA'))

class IcInstance:
    @staticmethod
    def build(cfg: Config) -> None:
        """Aggregate initial inventory from multiple instances into a single representative instance."""
        log.info("Aggregating initial inventory data...")
        total_instances = 0
        on_hand_inventory: typing.Dict[int, float] = {}
        
        for filename in os.listdir(cfg.instance_path):
            if filename.startswith("ic_"):
                file_path = os.path.join(cfg.instance_path, filename)
                total_instances += 1
                with open(file_path, "r") as file:
                    data = json.load(file)
                    for product in data.get("products", []):
                        product_id = product["id"]
                        on_hand_inventory[product_id] = on_hand_inventory.get(product_id, 0) + product["on_hand_inventory"]
        
        if total_instances == 0:
            log.warning("No initial condition instances found!")
            return
        
        # Compute average inventory per product
        on_hand_inventory = {key: np.floor(value / total_instances) for key, value in on_hand_inventory.items()}
        
        output_data = {"config": {"instance_name": "ic_instance"}, "products": [{"id": key, "on_hand_inventory": value} for key, value in on_hand_inventory.items()]}
        
        # Save the aggregated initial condition instance
        output_path = os.path.join(cfg.instance_path, "ic_instance.json")
        with open(output_path, "w") as json_file:
            json.dump(output_data, json_file, indent=4)
        
        log.info(f"Aggregated initial inventory saved to {output_path}")


class StaticInstance:
    @staticmethod
    def build(cfg: Config) -> None:
        """Aggregate demand data from multiple static instances into a single representative instance."""
        log.info("Aggregating demand data...")
        total_instances = 0
        demand_data: typing.Dict[str, float] = {}
        
        for filename in os.listdir(cfg.instance_path):
            if filename.startswith("static_"):
                file_path = os.path.join(cfg.instance_path, filename)
                total_instances += 1
                with open(file_path, "r") as file:
                    data = json.load(file)
                    for key, value in data.get("demand_data", {}).items():
                        demand_data[key] = demand_data.get(key, 0) + value
        
        if total_instances == 0:
            log.warning("No static instances found!")
            return
        
        # Compute average demand per product-week
        demand_data = {key: np.floor(value / total_instances) for key, value in demand_data.items()}
        
        # Load the structure from an existing instance
        static_template_path = os.path.join(cfg.instance_path, "static_instance_0.json")
        if not os.path.exists(static_template_path):
            log.error("Template static_instance_0.json not found!")
            return
        
        with open(static_template_path, "r") as file:
            static_data = json.load(file)
        static_data["demand_data"] = demand_data
        
        # Save the aggregated static instance
        output_path = os.path.join(cfg.instance_path, "static_instance.json")
        with open(output_path, "w") as file:
            json.dump(static_data, file, indent=4)
        
        log.info(f"Aggregated static demand data saved to {output_path}")


def main() -> None:
    """Main function to generate the aggregated instances."""
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    cfg = Config()
    
    log.info("Starting instance aggregation...")
    IcInstance.build(cfg)
    StaticInstance.build(cfg)
    log.info("Instance aggregation completed successfully!")


if __name__ == '__main__':
    main()
