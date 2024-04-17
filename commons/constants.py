import os

PWD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# logging
LOG_PADDING_WIDTH = 60
LOG_PADDING_SYMBOL = '*'

# training constants
THRESHOLD = 0.0001
MODEL_CONVERGENCE_COUNT = 10
MODEL_SAVE_COUNT = 10

# gcs project details
PROJECT = "cprtqa-datascience-sp1"

# gcs bucket details
GCS_BUCKET = "cprt_u2net_data"
TRAINING_FOLDER = "training"
PRETRAINED_MODEL_FOLDER = "pretrained_model"
BEST_MODEL_FOLDER = "best_model"
MODEL_CHECKPOINT = "model_checkpoint"
FINAL_MODEL = "final_model"

# gcs vertex ai training compute location
LOCATION = "us-west4"
