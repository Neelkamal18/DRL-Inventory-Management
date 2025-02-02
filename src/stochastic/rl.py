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
from rl_algo import *

log = logging.getLogger(__name__)

class Config(typing.NamedTuple):
    num_instances = 200
    saa_instance_path: str = os.path.join(os.path.dirname(__file__), '..', 'SAA')
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
    def build(cls, cfg: Config, initial_condition_instance_path, static_instance_path):
        """
        1. Read the static_instance.json file.
        2. Collect all the static data:
            - P, T, D, L, S, V, Vmax, R, F, H, G
        """
        with open(static_instance_path, 'r') as file:
            static_instance_data = json.load(file)

        product_ids = [product['id'] for product in static_instance_data.get('products', [])]
        week_number = [week for week in range(static_instance_data.get('config', {})['num_time_points'])]
        demand = static_instance_data.get('demand_data',{})
        lead_time = static_instance_data['config']['lead_time_in_weeks']
        safety_stock_product = {product['id']: math.floor(product['safety_stock_in_weeks']) for product in static_instance_data.get('products', [])}
        volume_product = {product['id']: product['volume'] for product in static_instance_data.get('products', [])}
        volume_max = static_instance_data['config']['container_volume']
        ramping_units = static_instance_data.get('ramping_factor', {})
        F = static_instance_data.get('fail_demand_cost', {})
        H = static_instance_data.get('overstocking_cost', {})
        G = static_instance_data.get('reward_recommended_stock', {})
        gamma_params = static_instance_data.get('gamma_params',{})

        """
        1. Read the ic_instance.json file.
        2. Collect all initial condition info.
             - i_p,0
        """
        with open(initial_condition_instance_path, 'r') as file:
            ic_data = json.load(file)

        initial_inventory = {product['id']: product['on_hand_inventory'] for product in ic_data.get('products', [])}

        """
        1. Create rest of the model parameters.
            - N, A, B
        """
        num_containers = 5
        large_constant_1 = 10000
        large_constant_2 = 10000

        return cls(config = cfg,
            P = product_ids,
            T = week_number,
            D = demand,
            L = lead_time,
            S = safety_stock_product,
            V = volume_product,
            Vmax = volume_max,
            R = ramping_units,
            F = F,
            H = H,
            G = G,
            init_inv = initial_inventory,
            N = num_containers,
            A = large_constant_1,
            B = large_constant_2,
            gamma_params = gamma_params
            )

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s',
                        level=logging.INFO)
    
    cfg = Config()
    log.info('Starting data read from both json files for static data and initial conditions.')
    for i in range(cfg.num_instances):
        ic_filename = os.path.join(cfg.saa_instance_path, f"ic_instance_{i}.json")
        static_filename = os.path.join(cfg.saa_instance_path, f"static_instance_{i}.json")
        data = Data.build(cfg, ic_filename, static_filename)
        log.info('Data read complete.')
    
        log.info('Setting up reinforcement learning states, actions and environment.')
        environment = ScmEnv(data)
        log.info('Environment setup complete.')
        algorithms = RLAlgorithms(environment, cfg)
        #Use this to make prediction using the save reinforcement learning model. Comment out otherwise.
        algorithms.algorithm_predict()
    
    """
    log.info('running reinforcement learning algorithms using stable baselines.')
    # --------Register your custom environment
    # --------gym.register(id='ScmEnv-v0', entry_point=ScmEnv(data))
    algorithms = RLAlgorithms(environment, cfg)

    log.info('checking environment using stable baselines3')
    algorithms.checkenv()

    #Use this to train the agent using reinforcement learning. Comment out otherwise.
    #algorithms.algorithm_train()
    """
    
    
if __name__ == '__main__':
    main()
