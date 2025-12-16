#!/bin/bash

# Generating test.txt, val.txt and testval.txt files based on the dataset
python3 run/generate_splits.py

# Running repo's data processing script
cd PointPillars || exit
python3 pre_process_kitti.py --data_root /workspace/carla_kitti

# Running the training pipeline
python3 train.py --data_root /workspace/carla_kitti
