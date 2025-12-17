# Project Run Guide

This document explains how to correctly start and run the full project (CARLA server + client/runtime).

## Prerequisites

- You have access to the VNC environment.
- CARLA is available at: `/shared/carla`
- Two Python environments are used:
  - **Python 3.8** (for the CARLA server)
  - **Python 3.12 venv** at `/home/shared/3_12_jupyter/bin/activate` (for the runtime client)

---

## How to Run the Project

### 1) Start CARLA (VNC)

Open a terminal in VNC:

```bash
cd /shared/carla
./CarlaEU4.sh
```
This will start carla.

## 2) In your IDE (pycharm):
- go to `RL/carla_server.py`
- switch to Python Interpreter 3.8
- Run this file

## 3) On the VNC
- create a new tab
```bash
source /home/shared/3_12_jupyter/bin/activate
cd /home/shared/3_12_jupyter/bin/
python3 -m runtime.main
```

## 4) Any shell
```shell
sudo docker compose up -d pointpillars
```

With these steps you can run our project and see the RL agent, object detection, lidar, and distance to vehicle in front
