# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

###  CONTROLS  (non-tunable)  ###

# general
TYPE_OF_RUN = train  # train, test, test_episodes, render
LOAD_MODEL_FROM = None
SAVE_MODELS_TO = models/new_wmg_flat_babyai.pth

# worker.py
ENV = BabyAI_Env
ENV_RANDOM_SEED = randint  # Use an integer for deterministic training.
AGENT_RANDOM_SEED = 1
REPORTING_INTERVAL = 10000
TOTAL_STEPS = 1600000
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
OBS_ENCODER = Flat
BINARY_REWARD = True

###  HYPERPARAMETERS  (tunable)  ###

# A3cAgent
A3C_T_MAX = 6
LEARNING_RATE = 2.5e-05
DISCOUNT_FACTOR = 0.5
GRADIENT_CLIP = 256.0
ENTROPY_TERM_STRENGTH = 0.02
ADAM_EPS = 1e-8
REWARD_SCALE = 16.0
WEIGHT_DECAY = 0.

# WMG
WMG_MAX_OBS = 0
WMG_MAX_MEMOS = 16
WMG_MEMO_SIZE = 64
WMG_NUM_LAYERS = 2
WMG_NUM_ATTENTION_HEADS = 16
WMG_ATTENTION_HEAD_SIZE = 24
WMG_HIDDEN_SIZE = 16
AC_HIDDEN_LAYER_SIZE = 512
