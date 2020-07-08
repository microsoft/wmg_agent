# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_gru_flat_babyai.pth

# worker.py
ENV = BabyAI_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 10000
TOTAL_STEPS = 2000000
ANNEAL_LR = False

# A3cAgent
AGENT_NET = GRU_Network

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = True
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Flat
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# A3cAgent
A3C_T_MAX = 4
LEARNING_RATE = 4e-05
DISCOUNT_FACTOR = 0.9
GRADIENT_CLIP = 512.0
ENTROPY_TERM_STRENGTH = 0.02
ADAM_EPS = 1e-12
REWARD_SCALE = 2.0
WEIGHT_DECAY = 0.

# RNNs
NUM_RNN_UNITS = 96
OBS_EMBED_SIZE = 512
AC_HIDDEN_LAYER_SIZE = 4096
