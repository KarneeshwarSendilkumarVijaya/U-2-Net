#!/bin/bash

# Add the root directory to the PYTHONPATH
export PYTHONPATH=${APP_DIR}:${PYTHONPATH}
echo "Training Epoch value passed: $epoch_num"
echo "Model save frequency passed: $save_frq"
# Run the u2net_train.py script
python service/u2net_train.py --epoch_num $epoch_num --save_frq $save_frq