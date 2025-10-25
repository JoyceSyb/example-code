# configs/config.py

import os

# Constants Definition
EPSILON = 1e-4
START_LEARNING_RATE_TEXT_ENCODER = 1e-5
START_LEARNING_RATE_GENERATOR = 1e-4
MAX_SEQ_LEN = 20
FREEZE_LAYERS = 6
MAX_LENGTH_PREDICTION = 80 
MAX_NUM_FRAMES = 80 


# Data Paths
TRAIN_FRENCH_FOLDER_2D = "/data/zmo/Swisspose/swissubase_2569_1_0/data/train_data_french_2d"
TEST_FRENCH_FOLDER_2D = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_french_2d"
TRAIN_GERMAN_FOLDER_2D = "/data/zmo/Swisspose/swissubase_2569_1_0/data/train_data_german_2d"
TEST_GERMAN_FOLDER_2D = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_german_2d"

# Training Parameters
BATCH_SIZE = 16
NUM_WORKERS = 2
MAX_EPOCHS = 300
TOTAL_STEPS = 30000 
LR_TEXT_ENCODER = START_LEARNING_RATE_TEXT_ENCODER
LR_GENERATOR = START_LEARNING_RATE_GENERATOR
LAMBDA_LENGTH = 0.5

# Diffusion Parameters
BETA_START = 1e-4
BETA_END = 0.02
ETA = 0.0

# Visualization Parameters
ORIGINAL_POSES_DIR = "/data/zmo/Swisspose/swissubase_2569_1_0/data/test_data_2d"
PREDICTED_POSES_DIR = "/data/zmo/Swisspose/Swisspose_project/test_result/"
OUTPUT_GIFS_DIR = "/data/zmo/Swisspose/Swisspose_project/test_result/output_gifs"
MAX_SAMPLES_VISUALIZATION = 20  # Set to None to process all files
GIF_FPS = 10
SHOW_AXES = True


# Checkpoint Paths
CHECKPOINT_DIR_BEST = '/data/zmo/Swisspose/Swisspose_project/best_model'
CHECKPOINT_DIR_LAST = '/data/zmo/Swisspose/Swisspose_project/checkpoints'

# Logger
LOG_DIR = "tb_logs"
LOG_NAME = "pose_generation_model"

# Others
TEST_RESULTS_DIR = '/data/zmo/Swisspose/Swisspose_project/test_result'
VALIDATION_SPLIT = 0.1 
