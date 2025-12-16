# 1. Autonomous Driving

# 2. Project Structure


# 3. Development setup
## 3.1. IDE
### 3.1.1. Install python-carla for virtual environment.
Downloading existing wheels for carla api here: https://www.wheelodex.org/projects/carla/wheels/carla-0.9.16-cp312-cp312-win_amd64.whl/
Make sure you select the correct version and platform.

*Example for Windows 3.12*
1) Create a new virtual environment on Python 3.12 
2) Download carla-0.9.16-cp312-cp312-win_amd64.whl
3) In pycharm, open terminal, go to /Downloads
4) Run `pip install carla-0.9.16-cp312-cp312-win_amd64.whl`

## 3.2. Carla simulation

## 3.3. Jupyter Notebook
### 3.3.1. Execute locally
Open a jupyter notebook file (`.ipynb` extension).
In the top right you can click `managed jupyter server: auto-start`, then `Configure Jupyter server`.

In the pop-up:
- Make sure you're on `Use managed server`
- Make sure you select the right interpreter (where carla is installed)

Then just run the code.

### 3.3.2. Execute remotely (VM)
https://uantwerpen.sharepoint.com/:w:/r/sites/I-Distributed-Artificial-Intelligence-Project-Group-2/Gedeelde%20documenten/Setup/Ssh%20connectie%20instellen.docx?d=wc48aeb1e13c24be69550ce7cd44ce874&csf=1&web=1&e=sXaHOt


# 4. Reinforcement Learning

## 4.1 CARLA Gym Environment

This repository contains a custom Gymnasium environment, named CarlaEnv, designed for training Deep Reinforcement Learning (DRL) agents in the CARLA Simulator.

The environment is specialized for Adaptive Cruise Control (ACC). The RL agent controls longitudinal acceleration (using Throttle and Brake), while lateral control (Steering) is handled automatically by a PID-based Local Planner.

### 4.1.1 **Key Features**

*   **Task:** Longitudinal control (Throttle/Brake) to follow traffic safely and efficiently.
    
*   **Automatic Map Switching:** Rotates through Town01, Town02, Town03, and Town04 every 25 episodes to prevent the agent from overfitting to a specific road topology.
    
*   **Dynamic Traffic:** Spawns random traffic (between 15 and 50 vehicles depending on map size) managed by CARLA's Traffic Manager.
    
*   **Remote Capable:** The environment is decorated with Pyro4 features, allowing it to run in a separate process or machine from the training loop.
    
*   **Robust Reward Function:** Features a "Proportional ACC" reward structure with elastic safety buffers and comfort penalties.
    
*   **Visualization:** Built-in HUD rendering and 3D bounding box visualization for debugging purposes.
    

### 4.1.2 Prerequisites

To use this environment, you need the CARLA Simulator (tested on versions compatible with the provided Python API) and Python 3.7 or higher.

You will need to install the following Python libraries: gymnasium, numpy, carla, and pyro4. 

### 4.1.3 Environment Specifications

#### 4.1.3.1 Action Space:
The action space is a continuous box containing a single value ranging from -1.0 to 1.0.
Values from -1.0 to 0.0 represent brake.
Values from 0.0 to 1.0 represent throttle.

#### 4.1.3.2 Observation Space: 
The state is a 1D vector with 35 dimensions containing ego-state and perception data:

* **Future Waypoints (30 dimensions):** Relative (x, y) coordinates of the next 15 path waypoints.
        
*   **Speed (1 dimension):** Ego vehicle speed in meters per second.
        
*   **Acceleration (1 dimension):** Ego vehicle acceleration in meters per second squared.
        
*   **Steering Command (1 dimension):** The steering angle calculated by the Local Planner (proprioception).
        
*   **Distance Ahead (1 dimension):** Ground truth distance to the leading vehicle (capped at 50 meters).
        
*   **Cross Track Error (1 dimension):** Distance deviation from the center lane.

#### 4.1.3.3 Reward Function: 
The reward function is designed for smooth, safe, and efficient driving:
    
*   **Collision:** A terminal penalty of -200 is applied.
        
*   **Target Speed (Proportional ACC):** If blocked by a lead car, the target speed scales linearly based on distance (closer equals slower). If the road is clear, the target speed is the speed limit.
        
*   **Safety Buffer:** Maintains an "elastic buffer" calculated as a minimum of 8 meters plus a 2-second gap based on speed. A high proximity penalty is applied if the distance drops below 4.0 meters to prevent bumper-hugging.
        
*   **Efficiency:** A Gaussian reward is given for matching the target speed, plus a bonus for progress when the road is clear.
        
*   **Comfort:** Penalties are applied for high Lateral or Longitudinal G-forces (Jerk).
        
*   **Goal:** A reward of +50 is given for reaching the destination.
        
*   _Note: Shaping rewards are clipped between -3 and 3 before adding the terminal goal reward._
        

### 4.1.4 Constraints

*   **Synchronous Mode:** The environment forces CARLA into Synchronous mode with a fixed time step of 0.05 seconds (20 FPS).
    
*   **Ego Vehicle:** The default vehicle used is the Tesla Model 3.
    
*   **Spawn Failures:** If the map is too crowded, spawning the ego vehicle might fail. The system attempts to retry 10 times before raising an error.

# 4.2 Carla server

This script acts as a network bridge, exposing the local CarlaEnv Gymnasium environment to remote clients via Pyro4. This decouples the simulation physics from the training logic, enabling distributed training.

## 4.2.1 Key Features

*   **Remote Hosting:** Exposes CarlaEnv as a network object, allowing clients to call step() and reset() remotely.
    
*   **Integrated Name Server:** Automatically starts a Pyro4 Name Server on port 9090 (background thread) for object discovery.
    
*   **Spectator Camera:** Automatically updates the simulator window's camera to follow the ego vehicle (3rd person view) every tick.
    
*   **Robust Connection:** Uses a 120s timeout to handle heavy map loading operations without disconnecting.
    

## 4.2.2 Configuration

*   **CARLA Host:** localhost : 2000
    
*   **Name Server:** 0.0.0.0 : 9090
    
*   **Object ID:** carla.environment
    

## 4.2.3 Usage

1.  Start **CARLA Simulator** (CarlaUE4.sh).
    
2.  Run this script. It will initialize the environment and wait for connections.
    
3.  Run your training script, which should locate the object carla.environment via the Name Server on port 9090.

# 4.3 CARLA Training

This script orchestrates the Deep Reinforcement Learning (DRL) training process for an autonomous driving agent in the CARLA simulator. It uses the Twin Delayed DDPG (TD3) algorithm from Stable Baselines 3 and integrates with Weights & Biases (WandB) for experiment tracking.

## 4.3.1 Key Features

*   **Robust Crash Recovery:** Automatically detects existing checkpoints and Replay Buffers. If a previous run ID is provided, it resumes training exactly where it left off, ensuring no data loss if the simulator crashes.
    
*   **Remote Environment:** Connects to a RemoteCarlaEnv (via Pyro4), allowing the training logic to run independently from the heavy simulator process.
    
*   **Experiment Tracking:** Fully integrated with WandB to log metrics, hyperparameters, and system stats in real-time.
    
*   **Emergency Saves:** Includes exception handling to save the model and replay buffer immediately if an error (like a simulator disconnect) or a keyboard interrupt occurs.
    

## 4.3.2 Configuration

*   **Algorithm:** TD3 (Twin Delayed Deep Deterministic Policy Gradient)
    
*   **Policy:** MLP (Multi-Layer Perceptron)
    
*   **Hyperparameters:**
    
    *   Learning Rate: 3e-4
        
    *   Buffer Size: 200,000 transitions
        
    *   Batch Size: 256
        
    *   Action Noise: Normal distribution (Sigma=0.1)
        
*   **Total Timesteps:** 200,000 (Configurable)
    
*   **Checkpoint Frequency:** Every 25,000 steps
    

## 4.3.3 Usage

1.  **Start the CARLA Pyro Server** (see separate server documentation).
    
2.  **Configure Parameters:** Edit the if \_\_name\_\_ == '\_\_main\_\_': block to set:
    
    *   TOTAL\_TIMESTEPS: Duration of training.
        
    *   PREVIOUS\_RUN\_ID: Set to None for a fresh run, or paste a WandB Run ID (e.g., "z4h8htc5") to resume.
        
3.  **Run:** Execute the script. It will automatically initialize WandB, connect to the environment, and begin (or resume) training.
    

**File Structure**

*   ./models/{run\_id}/: Stores periodic checkpoints (.zip models and .pkl replay buffers).
    
*   ./runs/{run\_id}/: Stores TensorBoard logs.
    

**Resume Logic**

The script checks the ./models/{run\_id} directory.

*   **Found:** Loads the latest model and replay buffer. Training continues from the saved timestep.
    
*   **Not Found:** Initializes a fresh TD3 model and starts training from step 0.