# Project Overview

This document provides a comprehensive overview of the MimicKit project structure, core modules, algorithm logic, and data flow.

## 1. Project Structure

The project is organized into the following key directories:

- **`mimickit/`**: Contains the core source code.
  - **`anim/`**: Character animation and kinematics (e.g., `KinCharModel`, `MotionLib`).
  - **`engines/`**: Abstraction layer for physics simulators (Isaac Gym, Isaac Lab, Newton).
  - **`envs/`**: Reinforcement learning environments (e.g., `DeepMimicEnv`, `AMPAgent`).
  - **`learning/`**: RL agents and algorithms (e.g., `PPOAgent`, `BaseAgent`, networks).
  - **`util/`**: Utilities for logging, math, and argument parsing.
  - **`run.py`**: The main entry point for training and testing.
- **`data/`**: Configuration files and assets.
  - **`agents/`**: Agent configurations (YAML).
  - **`envs/`**: Environment configurations (YAML).
  - **`engines/`**: Engine configurations (YAML).
- **`args/`**: Command-line argument files for reproducing experiments.
- **`docs/`**: Documentation files.

## 2. Core Modules

### 2.1 Engines (`mimickit/engines/`)
This module provides a unified interface for different physics simulators. The `Engine` class abstracts low-level simulation details, allowing the rest of the codebase to be agnostic to the backend (Isaac Gym, Isaac Lab, or Newton). Configuration is loaded from `data/engines/*.yaml`.

### 2.2 Environments (`mimickit/envs/`)
Environments define the task and interaction logic.
- **`BaseEnv`**: The abstract base class defining the standard RL interface (`reset`, `step`, `get_obs_space`, `get_action_space`).
- **`CharEnv`**: Extends `BaseEnv` for physics-based character tasks.
- **`DeepMimicEnv`**: A specific implementation for motion imitation tasks. It calculates rewards based on the tracking error between the simulated character and the reference motion (pose, velocity, root position, etc.).
- **Factory Pattern**: `env_builder.py` constructs environments based on the config file.

### 2.3 Agents (`mimickit/learning/`)
Agents encapsulate the learning algorithms.
- **`BaseAgent`**: Manages the training loop (`train_model`), testing loop (`test_model`), experience buffer, and model saving/loading.
- **`PPOAgent`**: Implements the Proximal Policy Optimization algorithm. It handles:
  - **Action Decision**: Sampling actions from the policy network.
  - **Value Estimation**: Using a critic network.
  - **Update**: Calculating advantages and updating network parameters.
- **Configuration**: Agent behavior (network structure, learning rate, clip ratio) is fully defined in `data/agents/*.yaml`.

### 2.4 Animation (`mimickit/anim/`)
Handles character kinematics and motion data.
- **`KinCharModel`**: Computes forward kinematics, joint rotations, and mappings between degrees of freedom (DoF) and rotation representations (quaternions, axis-angles).
- **`MotionLib`**: Loads and manages motion capture data, providing reference poses for imitation learning.

## 3. Data Flow

The data flow during a typical training session (initiated via `mimickit/run.py`) is as follows:

1.  **Initialization**:
    - `run.py` parses arguments and loads configurations.
    - `build_env()` creates the environment (e.g., `DeepMimicEnv`) and the underlying physics engine.
    - `build_agent()` creates the agent (e.g., `PPOAgent`) and initializes neural networks.

2.  **Interaction Loop (Sampling)**:
    - The Agent receives the current observation (`obs`) from the Env.
    - `Agent._decide_action(obs)` computes the action (`a`) using the policy network.
    - `Env.step(a)` executes the action in the physics engine.
    - The Env calculates the reward (`r`) and next observation (`next_obs`).
    - The tuple `(obs, a, r, next_obs, done)` is stored in the `ExperienceBuffer`.

3.  **Training Loop (Optimization)**:
    - Periodically, the Agent samples a batch of data from the `ExperienceBuffer`.
    - `PPOAgent` calculates the surrogate loss, value loss, and entropy.
    - The optimizer updates the network weights via backpropagation.
    - Logs (reward, loss, episode length) are recorded via `TensorBoard` or `WandB`.

4.  **Evaluation**:
    - Periodically, the Agent switches to `TEST` mode to evaluate performance without exploration noise.

## 4. Key Algorithms

- **DeepMimic**: A motion imitation approach where the reward function encourages the character to minimize the difference between its state and a reference motion frame.
- **PPO (Proximal Policy Optimization)**: The primary RL algorithm used for policy optimization, balancing sample efficiency and stability.
