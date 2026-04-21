#!/bin/bash

# ATTN-FIQA Attention Map Visualization

# Configuration
MODEL_PATH="../pretrained/"
MODEL_NAME="vits_wf4m_adaface.pt"
BACKBONE="vits"
GPU_ID=0
IMAGE_DIR="../data/test_images"
OUTPUT_FILE="../results/attention_visualization/results.png"
IMAGE_PATTERN="*.jpg"
MAX_IMAGES=25
TITLE="ATTN-FIQA Attention Visualization"

echo "=========================================="
echo "Attention Map Visualization"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Backbone: ${BACKBONE}"
echo "  GPU: ${GPU_ID}"
echo "  Image Directory: ${IMAGE_DIR}"
echo "  Output File: ${OUTPUT_FILE}"
echo "  Max Images: ${MAX_IMAGES}"
echo ""

# Check if image directory exists
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "Error: Image directory not found: ${IMAGE_DIR}"
    exit 1
fi

# Count images
NUM_IMAGES=$(find "${IMAGE_DIR}" -name "${IMAGE_PATTERN}" | wc -l)
echo "Found ${NUM_IMAGES} images in directory"
echo ""

# Create output directory if needed
OUTPUT_DIR=$(dirname "${OUTPUT_FILE}")
mkdir -p "${OUTPUT_DIR}"

# Run visualization
echo "Running attention map visualization..."
python plot_attnfiqa.py \
    --image-dir "${IMAGE_DIR}" \
    --image-pattern "${IMAGE_PATTERN}" \
    --output-file "${OUTPUT_FILE}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --backbone "${BACKBONE}" \
    --gpu-id ${GPU_ID} \
    --max-images ${MAX_IMAGES} \
    --title "${TITLE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Visualization Complete!"
    echo "=========================================="
    echo ""
    echo "Output saved to: ${OUTPUT_FILE}"
else
    echo ""
    echo "Error: Visualization failed!"
    exit 1
fi
