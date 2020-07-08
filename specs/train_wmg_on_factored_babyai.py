# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_wmg_factored_babyai.pth

# worker.py
ENV = BabyAI_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 1000
TOTAL_STEPS = 80000
ANNEAL_LR = False

# A3cAgent
AGENT_NET = WMG_Network

# WMG
V2 = False

# BabyAI_Env
BABYAI_ENV_LEVEL = BabyAI-GoToLocal-v0
USE_SUCCESS_RATE = True
SUCCESS_RATE_THRESHOLD = 0.99
HELDOUT_TESTING = True
NUM_TEST_EPISODES = 10000
OBS_ENCODER = Factored
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# A3cAgent
A3C_T_MAX = 6
LEARNING_RATE = 6.3e-05
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 512.0
ENTROPY_TERM_STRENGTH = 0.1
ADAM_EPS = 1e-12
REWARD_SCALE = 32.0
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 8
WMG_MEMO_SIZE = 32
WMG_NUM_LAYERS = 4
WMG_NUM_ATTENTION_HEADS = 2
WMG_ATTENTION_HEAD_SIZE = 128
WMG_HIDDEN_SIZE = 32
AC_HIDDEN_LAYER_SIZE = 2048
