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

## 3.2. Carla simulatie

## 3.3. Jupyter Notebook
### 3.3.1. Execute locally
Open a jupyter notebook file (`.ipynb` extension).
In the top right you can click `managed jupyter server: auto-start`, then `Configure Jupyter server`.

In the pop-up:
- Make sure you're on `Use managed server`
- Make sure you select the right interpreter (where carla is installed)

Then just run the code.

### 3.3.2. Execute remotely (VM)
