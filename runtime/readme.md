# Starting Guide
## 1. Shared memory
**Only do this if the folder does not exist yet** 
_Create:_
```shell
sudo mkdir /dev/shm/carla_shared
```
_Fix access_
```shell
sudo chmod -R 777 /dev/shm/carla_shared
```

## 2. Starting main script
Select correct venv
```shell
source /home/shared/3_8_jupyter/bin/activate
```
Go to runtime directory
```shell
cd /home/shared/project
```
Run runtime module
```shell
python3 -m runtime.main
```

## 2. Dockers
```shell
sudo docker compose up
```