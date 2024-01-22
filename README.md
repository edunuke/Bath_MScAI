# Introduction

This folder contains all code and dataset related to the the thesis for MSc. in AI.

The folder structure is as follows:

1. data:
   1. dataset_1: Synthetic dataset trained on TVAE from real data (main artifact for the thesis)
      1. raw: raw dataset as received
      2. processed: processed dataset after running model/1_dataprep.ipynb
   2. dataset_2: Secondary dataset from https://www.kaggle.com/datasets/parisrohan/credit-score-classification
      1. raw: raw dataset as downloaded
   
2. envs: virtual environment yaml file to run a replicated environment for running all code

3. imgs: images used during the thesis write up.

4. model: all code sits inside this directory
   1. logs: folder holds all logs from all agents experiment runs
   2. output: contains all serialized trained models
   3. utils:
      1. common.py : common code used across notebooks
      2. constants.py : constants to replicate sampled population
      3. doubledqn.py : personal implementation of double dqn extending stable baselines 3
      4. duelingdqn.py: personal implementation of dueling dqn extending stable baselines 3
      5. networks.py : personal implementation of different neural network arquitectures (feature extractors) for stable baselines 3
      6. riskenv.py : personal implementation of different environment for testing
   4. 1_dataprep.ipynb : all preprocessing of the raw data happens here.
   5. 2_modeling.ipynb : all modeling code for testing the agents and environment for dataset_1 (binary class)
   6. 3_modeling.ipynb : all modeling code for testing the agents and environment for dataset_2 (multi class dataset)


** All code is commented for better legibility **

