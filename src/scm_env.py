import random
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import gymnasium
from gymnasium import spaces    


class ScmEnv(gymnasium.Env):

    def __init__(self, data):
        """
        Initialize the SCM environment.
        """
        self.P = data.P  # List of product IDs
        self.N = data.N  # Max containers
        self.V = data.V  # Product volumes
        self.Vmax = data.Vmax  # Max volume per container
        self.R = data.R  # Ramping constraints
        self.S = data.S  # Safety stock weeks
        self.F = data.F  # Failure cost
        self.H = data.H  # Overstock cost
        self.G = data.G  # Reward for maintaining safety stock
        self.L = 3  # Number of weeks of future demands in observation
        self.T = data.T  # Episode length in weeks
        
        self.init_inv = data.init_inv  # Initial inventory
        self.inv_max = 5  # Max inventory
        self.gamma_params = data.gamma_params  # Demand distribution parameters
        self.D = {}  # Demand dictionary
        self.D_pred = data.D  # Predicted demands
        self.Dmax_p = 15  # Max demand per product (fixed value)
        
        self.train_same_instance_number = 1
        self.train_same_instance_counter = 0
        self.train_instance_type = 3
        
        self.observation_space = self.create_observation_space()
        self.action_space = self.create_action_space()
        
        self.current_obs = None
        self.previous_action = np.zeros(len(self.P), dtype=np.int32)
        self.week_num = 0
    
    def create_demands_episode(self):
        """
        Generate a demand scenario for the episode.
        
        Each product's demand is randomly sampled between 5 and 15 units 
        for the entire episode length plus the look-ahead period (L weeks).
        
        Returns:
            dict: A dictionary mapping (product_id, week) to demand values.
        """
        demand_data = {}

        for product_id in self.P:
            # Generate random demand for the full episode length + look-ahead period
            demand_sequence = [random.randint(5, 15) for _ in range(len(self.T) + self.L)]
            
            # Store demand for each week
            for week in range(len(self.T)):
                demand_key = f"{product_id}_{week}"
                demand_data[demand_key] = int(abs(demand_sequence[week]))

            # Ensure demand at the end of the time horizon is zero
            demand_data[f"{product_id}_{len(self.T)}"] = 0

        return demand_data
    
    def create_observation_space(self):
        """
        Define the observation space for the environment.
        
        The observation space consists of:
        - Inventory levels for each product (bounded by demand and safety stock).
        - Forecasted demand for each product in the next L weeks.
        - A binary indicator for shipping constraint violations.
        - A counter for ramping violations.

        Returns:
            gym.spaces.Box: The observation space, defining upper and lower bounds.
        """
        # Compute upper bounds for product inventory levels
        max_inventory_levels = np.array(
            [
                len(self.T) * self.Dmax_p * (1 + safety_stock) + self.inv_max
                for safety_stock in self.S.values()
            ],
            dtype=np.int32
        )

        # Compute upper bounds for demand forecasts
        max_demand_forecast = np.array(
            [self.Dmax_p for _ in self.S.values()], dtype=np.int32
        )

        # Upper bound for shipping constraint violations (binary: 0 or 1)
        max_shipping_violations = np.array([1], dtype=np.int32)

        # Upper bound for ramping violations (at most equal to the number of products)
        max_ramping_violations = np.array([len(self.P)], dtype=np.int32)

        # Concatenate all upper bounds to form the observation space
        upper_bounds = np.hstack(
            (max_inventory_levels, max_demand_forecast, max_shipping_violations, max_ramping_violations)
        )

        # Define observation space as a box with values ranging from 0 to upper bounds
        return spaces.Box(low=0, high=upper_bounds, dtype=np.int32)

    def create_action_space(self):
        """
        Define the action space for the environment.
        
        The action space represents the number of units of each product that can be ordered.
        Each action value is constrained by:
        - Maximum allowable demand per product (Dmax_p).
        - Safety stock multiplier.

        Returns:
            gym.spaces.Box: The action space, defining order quantity limits per product.
        """
        # Compute the upper bound for order quantities per product
        max_order_quantities = np.array(
            [self.Dmax_p * (1 + safety_stock) for safety_stock in self.S.values()],
            dtype=np.int32
        )

        # Define the action space as a continuous Box with values ranging from 0 to max order quantities
        return spaces.Box(low=0, high=max_order_quantities, dtype=np.int32)
    
    def prediction_reset(self, seed=None):
        """
        Reset environment to initial state/first observation to start new episode.
        """
        self.week_num = 0


        current_products = np.array([self.init_inv[p] for p in self.P], dtype=np.int32)


        self.D = self.D_pred
        for p in self.P:
            self.D[f"{p}_{len(self.T)}"] = 0


        demand_products = []
        for p in self.P:
            demand_products.append(self.D[f"{p}_{0}"])
        demand_products = np.array(demand_products, dtype=np.int32)
        
        shipping = np.array([0])
        ramping = np.array([0])

        self.current_obs = np.hstack((current_products, demand_products, shipping, ramping))
        return self.current_obs, {}
    
    def reset(self, seed=None):
        """
        Resets the environment to its initial state at the start of a new episode.

        This function:
        - Resets the week counter (`self.week_num`).
        - Determines if a new demand episode should be generated.
        - Randomly initializes inventory based on the training instance type.
        - Constructs the initial observation space, which consists of:
            - Current inventory levels for all products.
            - Demand for the first week.
            - Shipping and ramping violation indicators.

        Returns:
            tuple: (initial observation, empty dictionary for additional info)
        """
        self.week_num = 0  # Reset episode counter

        # Reset instance counter if needed
        if self.train_same_instance_counter == self.train_same_instance_number:
            self.train_same_instance_counter = 0

        # Generate new demand scenario if required
        if self.train_same_instance_counter == 0:
            self.D = self.create_demands_episode()
            
            # Assign a new random inventory state based on the training instance type
            self.current_products = np.array(
                [random.randint(0, self.inv_max) for _ in range(len(self.P))], dtype=np.int32
            )

            # Cycle through instance types (3 → 2 → 1 → 3)
            self.train_instance_type = 3 if self.train_instance_type == 1 else self.train_instance_type - 1

        self.train_same_instance_counter += 1  # Increment training instance counter

        # Construct the demand observation for the first week
        demand_products = np.array([self.D[f"{p}_0"] for p in self.P], dtype=np.int32)

        # Initialize shipping and ramping violation flags
        shipping = np.array([0])  # No shipping violations at the start
        ramping = np.array([0])  # No ramping violations at the start

        # Construct the initial observation space
        self.current_obs = np.hstack((self.current_products, demand_products, shipping, ramping))

        return self.current_obs, {}  # Return initial observation and empty dictionary for additional info
    
    def step(self, action):
        """
        Executes one step in the environment given an action.

        The function performs the following:
        1. Updates the inventory based on the action taken.
        2. Computes the reward based on:
            - Unmet demand penalty (F)
            - Overstock penalty (H)
            - Safety stock reward (G)
        3. Checks for constraint violations:
            - Shipping container volume constraints (Vmax)
            - Ramping constraints
        4. Returns the new observation, reward, episode termination status, and additional info.

        Args:
            action (np.array): The number of units ordered for each product.

        Returns:
            tuple: (new observation, reward, done flag, truncated flag, additional info)
        """
        action = np.floor(action)  # Ensure integer values

        # Initialize state variables
        next_obs = np.zeros(len(self.P) * 2 + 2, dtype=np.int32)  # Next observation state
        added_demand_units = np.zeros(len(self.P), dtype=np.int32)  # Newly added inventory
        unmet_demand_units = np.zeros(len(self.P), dtype=np.int32)  # Unmet demand

        penalty_F, penalty_H, reward_G = 0, 0, 0  # Reward components
        reward, shipping, ramping = 0, 0, 0  # Overall reward and constraint violations

        # Compute the next inventory state based on action and demand
        for index, element in np.ndenumerate(self.current_obs[:len(self.P)]):
            p_id = self.P[index[0]]
            added_demand_units[index] = action[index]

            if element + added_demand_units[index] > self.D[f"{p_id}_{self.week_num}"]:
                next_obs[index] = element + added_demand_units[index] - self.D[f"{p_id}_{self.week_num}"]
                unmet_demand_units[index] = 0
            else:
                next_obs[index] = 0
                unmet_demand_units[index] = self.D[f"{p_id}_{self.week_num}"] - (element + added_demand_units[index])

        # Compute rewards and penalties
        for index, element in np.ndenumerate(next_obs[:len(self.P)]):
            p_id = self.P[index[0]]
            
            # Unmet demand penalty
            if element == 0:
                penalty_F += -1 * (self.F[f"{p_id}"] * unmet_demand_units[index])
            else:
                if element > self.D[f"{p_id}_{self.week_num}"] * self.S[p_id]:  # Overstock penalty
                    penalty_H += -1 * (self.H[f"{p_id}"] * (element - self.D[f"{p_id}_{self.week_num}"] * self.S[p_id]))
                reward_G += self.G[f"{p_id}"] * element  # Safety stock reward

        reward = penalty_F + penalty_H + reward_G  # Compute final reward
        milp_reward = reward  # Store MILP-equivalent reward

        # Constraint Checks
        # 1. Shipping Constraint Violation (Vmax)
        total_volume = sum(action[index] * volume for index, volume in enumerate(self.V.values()))
        if total_volume > self.Vmax * self.N:
            reward -= 10
            shipping = 1

        # 2. Ramping Constraint Violation
        if self.week_num > 0:
            for curr, nxt, ramp in zip(self.current_obs[:len(self.P)], next_obs[:len(self.P)], self.R.values()):
                if abs(curr - nxt) >= ramp:
                    reward -= 10
                    ramping += 1

        # Update next week's demand in observation space
        for i, p in enumerate(self.P):
            next_obs[len(self.P) + i] = self.D[f"{p}_{self.week_num + 1}"]

        next_obs[-2] = shipping  # Shipping violation indicator
        next_obs[-1] = ramping  # Ramping violation indicator

        self.current_obs = next_obs  # Update the current observation state
        self.previous_action = action  # Store previous action for tracking

        # Check for episode termination
        self.week_num += 1
        episode_done = self.week_num >= len(self.T)

        return self.current_obs, reward, episode_done, False, {
            'milp_reward': milp_reward,
            'shipping_violated': shipping,
            'ramping_violated': ramping,
            'inventory': self.current_obs,
            'unmet': unmet_demand_units
        }

    def render(self, mode="human"):
        """
        Render the current state of the environment.

        This function is used to visualize the environment's state.
        Since visualization is not required in this case, it is left empty.
        
        Args:
            mode (str): The rendering mode (default: "human").
        
        Returns:
            None
        """
        pass

    def close(self):
        """
        Clean up environment resources.

        This function is useful for closing external resources, such as 
        graphical windows or background processes, if applicable.
        
        Since this environment does not use additional resources, it is left empty.

        Returns:
            None
        """
        pass

    def seed(self, seed=None):
        """
        Set the seed for the environment’s random number generator.

        This ensures reproducibility of results when running the environment multiple times.

        Args:
            seed (int, optional): The random seed to set. Default is None.

        Returns:
            list: A list containing the set seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        return [seed]




