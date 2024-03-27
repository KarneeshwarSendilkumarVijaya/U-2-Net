#!/bin/bash

export MODEL_NAME="u2net"
export DATASETS="xsmall"
export EPOCHS="'100000'"
export SAVE_FRQ="'2000'"
export BATCH_SIZE="'12'"
export CURRENT_TIME=$(date +"%Y%m%d%H%M%S")

envsubst < config.yaml > config_modified.yaml
envsubst < cloudbuild.yaml > cloudbuild_modified.yaml

gcloud builds submit --region=us-west4 --config cloudbuild_modified.yaml

gcloud ai custom-jobs create --config=config_modified.yaml --region=us-west4 --display-name="u2netTraining"

rm cloudbuild_modified.yaml
rm config_modified.yaml
