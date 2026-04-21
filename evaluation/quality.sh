#!/bin/bash

# Configuration
MODEL_PATH="../pretrained/"
MODEL_NAME="vits_wf4m_adaface.pt"
BACKBONE="vits"
GPU_ID=0
DATASETS="adience,lfw,calfw,cplfw,agedb_30,cfp_fp,XQLFW,IJBC"
DATA_DIR="../data/"
OUTPUT_DIR="../results/extracted_quality_scores"
BATCH_SIZE=32
COLOR_CHANNEL="BGR"

python getQualityScore.py \
    --model-path ${MODEL_PATH} \
    --model-name ${MODEL_NAME} \
    --backbone ${BACKBONE} \
    --gpu-id ${GPU_ID} \
    --datasets ${DATASETS} \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --color-channel ${COLOR_CHANNEL}
echo ""
