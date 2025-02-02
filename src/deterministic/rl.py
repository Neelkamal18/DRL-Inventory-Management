import os
import sys
import json
import logging
import typing
import math
import csv
import time

# Add src directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scm_env import ScmEnv
from rl_algo import RLAlgorithms

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    static_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'static_instance.json')
    initial_condition_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'ic_instance.json')
    result_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rl_results'))
    os.makedirs(result_path, exist_ok=True)
    model_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rl_model'))
    os.makedirs(model_path, exist_ok=True)

class Data(typing.NamedTuple):
    config: Config
    P: typing.List[int]
    T: typing.List[int]
    D: typing.Dict[str, int]
    L: int
    S: typing.Dict[int, float]
    V: typing.Dict[int, float]
    Vmax: float
    R: typing.Dict[str, int]
    F: typing.Dict[str, float]
    H: typing.Dict[str, float]
    G: typing.Dict[str, float]
    init_inv: typing.Dict[int, int]
    N: int
    A: int
    B: int
    gamma_params: typing.Dict[str, typing.Dict[str, float]]

    @classmethod 
    def build(cls, cfg: Config):
        """
        Reads static and initial condition JSON files and extracts data.
        """
        with open(cfg.static_instance_path, 'r') as file:
            static_instance_data = json.load(file)

        product_ids = [product['id'] for product in static_instance_data.get('products', [])]
        week_number = list(range(static_instance_data.get('config', {}).get('num_time_points', 0)))
        demand = static_instance_data.get('demand_data', {})
        lead_time = static_instance_data['config']['lead_time_in_weeks']
        safety_stock_product = {product['id']: math.floor(product['safety_stock_in_weeks']) for product in static_instance_data.get('products', [])}
        volume_product = {product['id']: product['volume'] for product in static_instance_data.get('products', [])}
        volume_max = static_instance_data['config']['container_volume']
        ramping_units = static_instance_data.get('ramping_factor', {})
        F = static_instance_data.get('fail_demand_cost', {})
        H = static_instance_data.get('overstocking_cost', {})
        G = static_instance_data.get('reward_recommended_stock', {})
        gamma_params = static_instance_data.get('gamma_params', {})

        with open(cfg.initial_condition_instance_path, 'r') as file:
            ic_data = json.load(file)

        initial_inventory = {product['id']: product['on_hand_inventory'] for product in ic_data.get('products', [])}

        return cls(config=cfg, P=product_ids, T=week_number, D=demand, L=lead_time, S=safety_stock_product,
                   V=volume_product, Vmax=volume_max, R=ramping_units, F=F, H=H, G=G, init_inv=initial_inventory,
                   N=5, A=10000, B=10000, gamma_params=gamma_params)

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s', level=logging.INFO)
    
    cfg = Config()
    log.info('Reading data from JSON files.')
    data = Data.build(cfg)
    log.info('Data read complete.')
    
    log.info('Initializing RL environment.')
    environment = ScmEnv(data)
    log.info('Environment setup complete.')
    
    log.info('Checking environment compatibility with Gymnasium.')
    if isinstance(environment, gymnasium.Env):
        log.info('ScmEnv is compatible with OpenAI Gym.')
    else:
        log.warning('ScmEnv is NOT compatible with OpenAI Gym.')
    
    log.info('Initializing RL algorithms.')
    algorithms = RLAlgorithms(environment, cfg)
    
    log.info('Checking environment using Stable Baselines3.')
    algorithms.checkenv()
    
    log.info('Training RL model.')
    algorithms.algorithm_train()
    
    log.info('Making predictions using the trained model.')
    algorithms.algorithm_predict()
    
if __name__ == '__main__':
    main()
