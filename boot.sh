#!/bin/bash

# Add the root directory to the PYTHONPATH
export PYTHONPATH=${APP_DIR}:${PYTHONPATH}
# Run the u2net_train.py script
python service/u2net_train.py --epochs=${EPOCHS} --save_frq=${SAVE_FRQ} --model_name=${MODEL_NAME} --datasets=${DATASETS} --batch_size=${BATCH_SIZE}
#python service/u2net_train.py
