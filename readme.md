# GAT-MF: Graph Attention Mean Field for Very Large Scale Multi-Agent Reinforcement Learning (Submitted to KDD'23)

## To run the grid-world experiments, you need:
- code/grid_simulator/: simultor for the grid-world experiment
- code/grid_networks.py: neural networks definition for the grid-world experiment
- code/grid_train.py: run this Python file to execute the training
- model/: trained models will be stored to this folder

## To run the real-world experiments, you need:
- data/: please download the required data (~10GB) from https://drive.google.com/drive/folders/1-68jPOd6NXVyiC1PWbo-9wrqiktOi4GT?usp=sharing and substitute this folder
- code/simulator/: simultor for the real-world experiment
- code/networks.py: neural networks definition for the real-world experiment
- code/constants.py: intrinsic constants for the real-world experiment
- code/data_range.py: data ranges for normalizations in the real-world experiment
- code/reward_fun.py: reward function definition for the real-world experiment
- code/util.py: util functions for the real-world experiment
- code/train.py: run this Python file to execute the training
- model/: trained models will be stored to this folder
