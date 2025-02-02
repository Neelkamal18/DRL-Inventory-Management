#using stable_baselines3 for RL algorithms
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from rl_zoo3 import linear_schedule

from scm_env import ScmEnv


from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the best training reward.

    This callback checks the mean training reward every `check_freq` steps
    and saves the model if it achieves a new highest reward.

    Args:
        check_freq (int): Frequency (in steps) to check for a new best model.
        log_dir (str): Path to the logging directory where models are saved.
        verbose (int, optional): Verbosity level (default: 1). 
                                 0 - No output, 1 - Info messages, 2 - Debug messages.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf  # Initialize best reward as negative infinity

    def _init_callback(self) -> None:
        """Initialize the callback by ensuring the save directory exists."""
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Check training reward periodically and save the model if it improves."""
        if self.n_calls % self.check_freq == 0:
            # Load training results
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Compute mean reward over the last 10,000 episodes
                mean_reward = np.mean(y[-10000:])

                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # Save model if the new mean reward is the best so far
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"New best model found! Saving to {self.save_path}")
                    self.model.save(self.save_path)

        return True

class RLAlgorithms:
    """
    Implements reinforcement learning algorithms for training and inference.
    """

    def __init__(self, environment, cfg, algo_name='PPO'):
        """
        Initializes the RL algorithm class.

        Args:
            environment (ScmEnv): The supply chain environment.
            cfg: Configuration object with paths for saving models and results.
            algo_name (str): The RL algorithm to use ('PPO', 'A2C', 'SAC').
        """
        self.P = environment.P  # Number of products
        self.T = environment.T  # Number of time steps per episode
        self.N = int(4_000_000 * len(self.T))  # Total training timesteps
        self.environment = environment  # Supply Chain Management (SCM) Environment
        self.algo_name = {'PPO': PPO, 'A2C': A2C, 'SAC': SAC}.get(algo_name, PPO)  # Select RL algorithm
        self.model_path = cfg.model_path  # Path to save RL model
        self.model = None
        self.result_path = cfg.result_path  # Path to save results

    def check_environment(self):
        """Check the environment for compatibility with Stable-Baselines3."""
        return check_env(self.environment)

    def train(self):
        """
        Train the RL model using the chosen algorithm.
        """
        self.environment = Monitor(self.environment, self.model_path)
        self.model = self.algo_name(
            "MlpPolicy",
            self.environment,
            ent_coef=0.01,
            verbose=0,
            tensorboard_log=self.model_path,
        )

        callback = SaveOnBestTrainingRewardCallback(check_freq=10_000, log_dir=self.model_path)
        self.model.learn(total_timesteps=self.N, progress_bar=True, callback=callback)

        # Plot training results
        plot_results([self.model_path], self.N, results_plotter.X_TIMESTEPS, "ScmEnv")
        plt.savefig(os.path.join(self.model_path, "training_reward_plot.png"))
        plt.close()

    def predict(self):
        """
        Run inference on a trained model to make predictions.
        """
        # Load the trained model
        self.model = self.algo_name.load(os.path.join(self.model_path, "best_model.zip"))

        # Reset environment and start prediction
        obs, info = self.environment.prediction_reset()
        episode_done = False
        optimal_actions = []
        shipping_violations = []
        ramping_violations = []
        milp_episode_reward = 0
        rl_episode_reward = 0

        while not episode_done:
            # Predict action using trained model
            action, _states = self.model.predict(obs, deterministic=True)
            action = np.floor(action)  # Ensure discrete order quantities
            optimal_actions.append(action.tolist())

            # Execute action in the environment
            obs, reward, episode_done, episode_truncated, info = self.environment.step(action)
            rl_episode_reward += reward
            milp_episode_reward += info['milp_reward']
            shipping_violations.append(info['shipping_violated'])
            ramping_violations.append(info['ramping_violated'])

        # Save results to CSV
        self.write_results_to_csv(rl_episode_reward, milp_episode_reward, optimal_actions, shipping_violations, ramping_violations)

        return optimal_actions

    def write_results_to_csv(self, reward, milp_reward, optimal_actions, shipping_violations, ramping_violations):
        """
        Save the RL results to CSV files.

        Args:
            reward (float): RL episode reward.
            milp_reward (float): MILP-equivalent reward.
            optimal_actions (list): List of optimal actions taken.
            shipping_violations (list): List of shipping constraint violations.
            ramping_violations (list): List of ramping constraint violations.
        """
        os.makedirs(self.result_path, exist_ok=True)

        # Write summary results
        result_file = os.path.join(self.result_path, "rl_results.csv")
        result_data = [reward, milp_reward]
        file_exists = os.path.isfile(result_file)

        with open(result_file, "a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Instance", "RL Reward", "MILP Reward"])
            file.seek(0, os.SEEK_END)
            instance_number = sum(1 for _ in file)
            writer.writerow([instance_number] + result_data)

        # Write detailed action results
        action_file = os.path.join(self.result_path, f"result_instance_action_{instance_number}.csv")
        with open(action_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time Period"] + self.P + ["Shipping Violations", "Ramping Violations"])
            for t in self.T:
                row_values = [t] + optimal_actions[t] + [shipping_violations[t], ramping_violations[t]]
                writer.writerow(row_values)

