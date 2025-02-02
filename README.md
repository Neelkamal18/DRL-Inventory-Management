# DRL-Inventory-Management

## 📌 Overview
This repository explores the application of **Deep Reinforcement Learning (DRL)** for **Inventory Management** in supply chains. It focuses on optimizing **safety stock levels** in a warehouse by leveraging **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm. 

Traditional inventory management approaches, such as **Binary Integer Programming (BIP)**, struggle with dynamic supply chain environments. This project transforms the inventory optimization problem into a **Markov Decision Process (MDP)** and applies **DRL techniques** to optimize order quantities while satisfying demand, shipping constraints, and operational limitations.

## Project Structure
```
📂 DRL-Inventory-Management
│── 📂 src                   # Source code directory
│   ├── 📂 deterministic     # Deterministic RL implementation
│   ├── 📂 stochastic        # Stochastic RL implementation
│   ├── 📂 other             # Supporting scripts
│── 📂 data                  # Dataset and demand forecasts
│── 📂 models                # Trained models and RL checkpoints
│── 📂 results               # Experimental results and performance comparisons
│── 📜 requirements.txt      # Python dependencies
│── 📜 README.md             # Project documentation
├── LICENSE                  # Open-source license
├── .gitignore               # Ignore unnecessary files
```

## Features
- **Formulates Inventory Planning as an MDP**
- **Uses PPO (Stable Baselines3) for training an RL agent**
- **Compares DRL-based inventory management with BIP solutions**
- **Implements constraints for demand fulfillment, shipping, and storage capacity**
- **Evaluates the model's performance using realistic demand scenarios**

## Installation
Ensure you have Python **3.8+** installed. Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Methodology
1. **Binary Integer Programming (BIP)**: Traditional optimization model.
2. **Markov Decision Process (MDP) Formulation**:
   - **State Space**: Inventory levels, demand forecasts, shipping constraints.
   - **Action Space**: Number of units ordered per time step.
   - **Reward Function**: Penalizes unmet demand and overstocking while rewarding optimal safety stock levels.
3. **Deep Reinforcement Learning**:
   - Uses **PPO** from Stable Baselines3 to train the agent.
   - The agent learns to optimize inventory levels over multiple episodes.
4. **Performance Evaluation**:
   - Compare BIP vs. PPO-based RL model.
   - Assess adaptability to changing demand and supply constraints.

## Running the Model
### 1️⃣ Train RL Agent
For **stochastic** inventory optimization:
```sh
python src/stochastic/rl.py
```
For **deterministic** inventory optimization:
```sh
python src/deterministic/rl.py
```

### 2️⃣ Evaluate Trained Model
For inventory evaluation:
```sh
python src/milp_evaluation.py
```

## Results
- **DRL-based inventory management adapts dynamically to demand fluctuations**.
- **PPO outperforms BIP in handling stochastic and dynamic environments**.
- **The model effectively balances order quantity, safety stock, and shipping constraints**.


## License
This project is licensed under the **MIT License**.

---


