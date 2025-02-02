import json
import logging
import typing
import random
import math
import csv
import time
from docplex.mp.model import Model
from docplex.mp.solution import *
from pathlib import Path

log = logging.getLogger(__name__)
    

class Config(typing.NamedTuple):
    """
    Configuration class for Sample Average Approximation (SAA) MILP.
    Defines file paths for input data and results.
    """

    BASE_DIR: Path = Path(__file__).resolve().parent.parent / 'SAA'
    RESULT_DIR: Path = Path(__file__).resolve().parent.parent / 'milp_results'

    static_instance_path: Path = BASE_DIR / 'static_instance.json'
    initial_condition_instance_path: Path = BASE_DIR / 'ic_instance.json'
    result_path: Path = RESULT_DIR

    # Ensure the result directory exists
    def __new__(cls):
        cls.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return super().__new__(cls)

import json
import typing
from pathlib import Path

class Data(typing.NamedTuple):
    """
    Data class for Sample Average Approximation (SAA) MILP optimization.

    Stores key input parameters such as:
    - Product details
    - Demand and inventory levels
    - Cost parameters
    - Safety stock, volume, and proportionality constraints
    """

    # Model configuration
    config: "Config"

    # Sets and indices
    P: list[int]  # Product IDs
    T: list[int]  # Time periods

    # Demand and Inventory Parameters
    D: dict[str, int]  # Demand over time
    init_inv: dict[int, int]  # Initial inventory levels
    L: int  # Lead time in weeks

    # Product-specific Constraints
    S: dict[int, float]  # Safety stock per product
    V: dict[int, float]  # Volume per product
    Vmax: float  # Maximum container volume

    # Cost Parameters
    F: dict[str, float]  # Cost of failing to meet demand
    H: dict[str, float]  # Cost of overstocking
    G: dict[str, float]  # Reward for maintaining recommended stock

    # Constraint Parameters
    R: dict[str, int]  # Ramping constraints (limit on order changes)
    alpha: float | int  # Proportionality cost penalty

    # Optimization Model Parameters
    N: int  # Number of containers available
    A: int  # Large constant for constraints
    B: int  # Large constant for constraints
    C: list[set[str]]  # Proportionality constraint groups

    @classmethod
    def build(cls, cfg: "Config") -> "Data":
        """
        Reads input data from JSON files and initializes the Data class.

        Steps:
        1. Read `static_instance.json` for model parameters.
        2. Read `ic_instance.json` for initial inventory levels.
        3. Compute additional model parameters (e.g., container constraints, proportionality groups).
        """

        # ---- Read Static Data ----
        with open(cfg.static_instance_path, "r") as file:
            static_data = json.load(file)

        config_data = static_data.get("config", {})
        products = static_data.get("products", [])

        # Extract model parameters
        product_ids = [p["id"] for p in products]
        time_periods = list(range(config_data.get("num_time_points", 0)))
        demand = static_data.get("demand_data", {})

        # Lead time
        lead_time = config_data.get("lead_time_in_weeks", 0)

        # Product constraints
        safety_stock = {p["id"]: p.get("safety_stock_in_weeks", 0) for p in products}
        volume = {p["id"]: p.get("volume", 0.0) for p in products}
        max_volume = config_data.get("container_volume", 0.0)

        # Cost parameters
        fail_cost = static_data.get("fail_demand_cost", {})
        overstock_cost = static_data.get("overstocking_cost", {})
        reward_stock = static_data.get("reward_recommended_stock", {})

        # Constraint parameters
        ramping_limit = static_data.get("ramping_factor", {})
        proportionality_cost = static_data.get("proportionality_cost", 1)

        # ---- Read Initial Inventory Data ----
        with open(cfg.initial_condition_instance_path, "r") as file:
            ic_data = json.load(file)

        initial_inventory = {p["id"]: p.get("on_hand_inventory", 0) for p in ic_data.get("products", [])}

        # ---- Additional Model Parameters ----
        NUM_CONTAINERS = 50  # Maximum number of shipping containers
        LARGE_CONSTANT_1 = 10_000  # Large constant for constraints
        LARGE_CONSTANT_2 = 10_000  # Another large constant for constraints

        # Define proportionality constraint groups
        proportionality_groups = [
            {p_id for p_id, stock in safety_stock.items() if stock == s}
            for s in set(safety_stock.values())
        ]

        # ---- Return Structured Data ----
        return cls(
            config=cfg,
            P=product_ids,
            T=time_periods,
            D=demand,
            L=lead_time,
            S=safety_stock,
            V=volume,
            Vmax=max_volume,
            R=ramping_limit,
            F=fail_cost,
            H=overstock_cost,
            G=reward_stock,
            alpha=proportionality_cost,
            init_inv=initial_inventory,
            N=NUM_CONTAINERS,
            A=LARGE_CONSTANT_1,
            B=LARGE_CONSTANT_2,
            C=proportionality_groups
        )


class OptimizationModel:
    """
    Defines and solves the Sample Average Approximation (SAA) MILP optimization model.

    This class:
    - Initializes the MILP model
    - Defines decision variables
    - Adds all necessary constraints
    - Defines the objective function
    - Optimizes the model using multi-threading and solver performance settings
    """
    def __init__(self, data, cfg):
        """
        Initializes model parameters from the given data and configuration.
        """
        self.P = data.P
        self.T = data.T
        self.D = data.D
        self.L = data.L
        self.S = data.S
        self.V = data.V
        self.Vmax = data.Vmax
        self.F = data.F
        self.H = data.H
        self.G = data.G
        self.alpha = data.alpha
        self.init_inv = data.init_inv
        self.N = data.N
        self.A = data.A
        self.B = data.B
        self.R = data.R
        self.C = data.C

        self.model = None
        self.solve_time = None
        self.result_path = cfg.result_path

    def build_model(self):
        """
        Constructs the MILP model, including decision variables, constraints, and the objective function.
        """
        self.model = Model()

        # ---- Define Decision Variables ----
        self.x = {(p, t): self.model.integer_var(lb=0, name=f'x_{p}_{t}') for p in self.P for t in self.T}
        #self.n = {t: self.model.integer_var(lb=0, name=f'n_{t}') for t in self.T}
        self.i = {(p, t): self.model.integer_var(lb=0, name=f'i_{p}_{t}') for p in self.P for t in self.T}
        self.u = {(p, t): self.model.integer_var(lb=0, name=f'u_{p}_{t}') for p in self.P for t in self.T}
        self.y = {(p, t): self.model.binary_var(name=f'y_{p}_{t}') for p in self.P for t in self.T}
        self.e = {(p, t): self.model.integer_var(lb=0, name=f'e_{p}_{t}') for p in self.P for t in self.T}
        self.z = {(p, t): self.model.binary_var(name=f'z_{p}_{t}') for p in self.P for t in self.T}
        self.o = {(p, t): self.model.integer_var(lb=0, name=f'o_{p}_{t}') for p in self.P for t in self.T}
        self.j = {(p, t): self.model.binary_var(name=f'j_{p}_{t}') for p in self.P for t in self.T}
        self.q = {(p, t): self.model.integer_var(lb=0, name=f'q_{p}_{t}') for p in self.P for t in self.T}
        self.k = {(p, t): self.model.binary_var(name=f'k_{p}_{t}') for p in self.P for t in self.T}
        self.m = {(p, t): self.model.binary_var(name=f'm_{p}_{t}') for p in self.P for t in self.T}
        self.sigma = {(p, idx, t): self.model.integer_var(lb=0, name=f'sigma_{p}_{idx}_{t}') for idx, c in enumerate(self.C) for p in c for t in self.T}

        # ---- Add Constraints ----
        self._add_inventory_constraints()
        self._add_demand_constraints()
        self._add_shipping_constraints()
        self._add_ramping_constraints()
        self._add_proportionality_constraints()
        self._add_variable_restrictions()

        # ---- Define Objective Function ----
        self._set_objective_function()
    
    def _add_inventory_constraints(self):
        """ Adds initial inventory constraints. """
        for p in self.P:
            self.model.add_constraint(
                self.i[(p, 0)] == self.init_inv[p],
                ctname=f"initial_inventory_constraint_{p}"
            )
    
    def _add_demand_constraints(self):
        """ Adds demand satisfaction and inventory balance constraints. """
        for p in self.P:
            for t in self.T:
                self.model.add_constraint(
                    self.x[(p, t)] + self.i[(p, t)] + self.u[(p, t)] == self.D[f"{p}_{t}"] + self.e[(p, t)],
                    ctname=f"demand_balance_{p}_{t}"
                )
                self.model.add_constraint(
                    self.u[(p, t)] <= self.A * self.y[(p, t)],
                    ctname=f"demand_constraint_{p}_{t}_1"
                )
                self.model.add_constraint(
                    self.e[(p, t)] <= self.A * self.z[(p, t)],
                    ctname=f"demand_constraint_{p}_{t}_2"
                )
                self.model.add_constraint(
                    self.y[(p, t)] + self.z[(p, t)] <= 1,
                    ctname=f"demand_constraint_{p}_{t}_3"
                )
                if t + 1 < len(self.T):
                    self.model.add_constraint(
                        self.e[(p, t)] == self.i[(p, t + 1)],
                        ctname=f"demand_constraint_{p}_{t}_4"
                    )
                self.model.add_constraint(
                    self.e[(p, t)] == self.S[p] * self.D[f"{p}_{t}"] + self.o[(p, t)] - self.q[(p, t)],
                    ctname=f"demand_constraint_{p}_{t}_5"
                )
                self.model.add_constraint(
                    self.o[(p, t)] <= self.B * self.j[(p, t)],
                    ctname=f"demand_constraint_{p}_{t}_6"
                )
                self.model.add_constraint(
                    self.q[(p, t)] <= self.B * self.k[(p, t)],
                    ctname=f"demand_constraint_{p}_{t}_7"
                )
                self.model.add_constraint(
                    self.j[(p, t)] + self.k[(p, t)] <= 1,
                    ctname=f"demand_constraint_{p}_{t}_8"
                )
                self.model.add_constraint(
                    self.j[(p, t)] + self.k[(p, t)] + self.m[(p, t)] == 1,
                    ctname=f"demand_constraint_{p}_{t}_9"
                )
    
    def _add_shipping_constraints(self):
        """ Adds shipping constraints based on volume limits. """
        for t in self.T:
            self.model.add_constraint(
                self.model.sum(self.V[p] * self.x[(p, t)] for p in self.P) <= self.Vmax * self.N,
                ctname=f"shipping_constraint_{t}"
            )
        """
            self.model.add_constraint(
                self.n[t] <= self.C,
                ctname=f"shipping_constraint_{t}"
            )
            self.model.add_constraint(
                self.model.sum(self.V[p] * self.x[(p, t)] for p in self.P) <= self.Vmax * self.n[t],
                ctname=f"shipping_constraint1_{t}"
            )
            self.model.add_constraint(
                0.98 * self.Vmax * self.n[t] <= self.model.sum(self.V[p] * self.x[(p, t)] for p in self.P),
                ctname=f"shipping_constraint2_{t}"
            )
        """
    
    def _add_ramping_constraints(self):
        """ Adds ramping constraints to limit changes in order quantities. """
        """
        for p in self.P:
            for t in self.T:
                if t > 0:
                    self.model.add_constraint(
                        self.x[(p, t)] - self.x[(p, t - 1)] <= self.R[f"{p}"],
                        ctname=f"ramping_constraint1_{p}_{t}"
                    )
                    self.model.add_constraint(
                        -self.R[f"{p}"] <= self.x[(p, t)] - self.x[(p, t - 1)],
                        ctname=f"ramping_constraint2_{p}_{t}"
                    )
        """
        
    def _add_proportionality_constraints(self):
        """ Adds proportionality constraints across grouped products. """
        """
        for idx, c in enumerate(self.C):
            for p in c:
                for t in self.T:
                    if p == next(iter(c)):
                        self.model.add_constraint(
                            self.sigma[(p, idx, t)] == 0,
                            ctname=f"proportionality_constraint1_{p}_{idx}_{t}"
                            #setting proportionality violation for the first product type in each category to zero since the comparision is w.r.t. first product type in group, for all weeks.
                        )
                    elif t == 0 and p != next(iter(c)):
                        self.model.add_constraint(
                            self.sigma[(p, idx, t)] == 0,
                            ctname=f"proportionality_constraint2_{p}_{idx}_{t}"
                            #setting proportionality violation for all product types for first week in each category to zero since initial inventory is predecided.
                        )
                    else:
                        self.model.add_constraint(
                            self.sigma[(p, idx, t)] >= self.i[(next(iter(c)), t)] - self.i[(p, t)],
                            ctname=f"proportionality_constraint3_{p}_{idx}_{t}"
                        )
                        self.model.add_constraint(
                            self.sigma[(p, idx, t)] >= self.i[(p, t)] - self.i[(next(iter(c)), t)],
                            ctname=f"proportionality_constraint4_{p}_{idx}_{t}"
                        )
        """ 
        
    def _add_variable_restrictions(self):
        """ Adds constraints to enforce variable bounds and types. """
        for p in self.P:
            for t in self.T:
                self.model.add_constraint(self.x[(p, t)] >= 0)
                self.model.add_constraint(self.i[(p, t)] >= 0)
                self.model.add_constraint(self.u[(p, t)] >= 0)
                self.model.add_constraint(self.e[(p, t)] >= 0)
                self.model.add_constraint(self.o[(p, t)] >= 0)
                self.model.add_constraint(self.q[(p, t)] >= 0)
                self.model.add_constraint(self.x[(p, t)].is_integer())
                self.model.add_constraint(self.i[(p, t)].is_integer())
                self.model.add_constraint(self.u[(p, t)].is_integer())
                self.model.add_constraint(self.e[(p, t)].is_integer())
                self.model.add_constraint(self.o[(p, t)].is_integer())
                self.model.add_constraint(self.q[(p, t)].is_integer())
                self.model.add_constraint(self.y[(p, t)].is_binary())
                self.model.add_constraint(self.z[(p, t)].is_binary())
                self.model.add_constraint(self.j[(p, t)].is_binary())
                self.model.add_constraint(self.k[(p, t)].is_binary())
                self.model.add_constraint(self.m[(p, t)].is_binary())

    def _set_objective_function(self):
        """
        Defines the objective function to minimize total costs.
        """
        objective_expr = (
            self.model.sum(self.F[f"{p}"] * self.u[p, t] for p in self.P for t in self.T) +
            self.model.sum(self.H[f"{p}"] * self.o[p, t] for p in self.P for t in self.T) -
            self.model.sum(self.G[f"{p}"] * self.m[p, t] * self.S[p] * self.D[f"{p}_{t}"] for p in self.P for t in self.T) -
            self.model.sum(self.G[f"{p}"] * (self.S[p] * self.D[f"{p}_{t}"] * self.k[p, t] - self.q[p, t]) for p in self.P for t in self.T) #+ 
            #self.model.sum(self.alpha*self.sigma[p, idx, t] for idx, c in enumerate(self.C) for p in c for t in self.T)
        )
        self.model.minimize(objective_expr)

    def optimize(self):
        """
        Optimizes the MILP model with solver performance enhancements.
        """
        self.model.parameters.threads = 4  # Utilize multi-core processing
        self.model.parameters.preprocessing.presolve = 1  # Enable presolve to simplify model
        self.model.parameters.mip.tolerances.mipgap = 0.01  # Set optimality gap for faster convergence

        start_time = time.time()
        self.model.solve()
        self.solve_time = time.time() - start_time

        if self.model.solution:
            log.info(f"Optimal solution found: {self.model.solution.get_objective_value()} in {self.solve_time:.2f} seconds.")
        else:
            log.warning("No optimal solution found.")
        
        totalSS = 0
        totalDemand = 0
        totalinv = 0
        unmet = 0
        for t in self.T:
            for p in self.P:
                totalSS +=self.S[p] * self.D[f"{p}_{t}"]
            print("Total SS: " + str(totalSS))
            totalSS=0
        for t in self.T:
            for p in self.P:
                totalDemand +=self.D[f"{p}_{t}"]
            print("Total Demand: " + str(totalDemand))
            totalDemand=0
        for t in self.T:
            for p in self.P:
                totalinv += self.model.solution.get_value(self.e[(p,t)])
            print("Total inventory: " + str(totalinv))
            totalinv = 0
        for t in self.T:
            for p in self.P:
                unmet += self.model.solution.get_value(self.e[(p,t)])
            print("Total unmet: " + str(totalinv))
            unmet = 0
        """
        for idx, c in enumerate(self.C):
            for p in c:
                for t in self.T:
                    print(self.sigma[p, idx, t], self.model.solution.get_value(self.sigma[p, idx, t]))
        """

    def write_to_csv(self):
        """
        Saves the results of the optimization to a CSV file.
        """
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        result_file = os.path.join(self.result_path, 'SAA_results.csv')
        result_data = [self.model.solve_status, self.model.solution.objective_value, self.solve_time]

        file_exists = os.path.isfile(result_file)
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Instance name", "Optimality Status", "Objective Function", "Solve Time"])
            writer.writerow(result_data)
        file.close()
        with open(result_file, "r") as file:
            no_lines = sum(1 for _ in file)
        # Write result data
        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([no_lines] + result_data)

        #need self.x[(p, t)], self.n[t]
        result_instance_file = os.path.join(self.result_path, 'SAA_result_instance_action.csv')
        with open(result_instance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time period"] + self.P)
            for t in self.T:
                row_values = [t]
                for p in self.P:
                     row_values.append(self.model.solution.get_value(self.x[p,t]))
                writer.writerow(row_values)


def main():
    """
    Executes the full optimization pipeline, including:
    - Loading configuration and data
    - Building and solving the MILP model
    - Logging key milestones
    - Writing results to CSV
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s--: %(message)s', level=logging.INFO)
    log.info('Initializing configuration and loading data.')
    cfg = Config()
    data = Data.build(cfg)
    log.info('Data successfully loaded.')
    
    log.info('Initializing and building optimization model.')
    model = OptimizationModel(data, cfg)
    model.build_model()
    log.info('Optimization model successfully built.')
    
    log.info('Starting optimization process.')
    model.optimize()
    log.info('Optimization completed.')
    
    log.info('Saving results to CSV.')
    model.write_to_csv()
    log.info('Results successfully saved.')
    
if __name__ == '__main__':
    main()
