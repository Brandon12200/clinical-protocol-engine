#!/bin/bash
# train_model.sh - Script to download data and train the model

# Default values
DATA_DIR="data/training"
OUTPUT_DIR="models/biobert_model"
CONFIG_FILE="training/config.json"
DATASET="jnlpba"
FORCE_DOWNLOAD=false

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 1: Download and process data
echo "=== Downloading and processing data ==="
python training/download_data.py --data_dir "$DATA_DIR" --datasets "$DATASET" --convert
if [ $? -ne 0 ]; then
    echo "Error: Data download failed!"
    exit 1
fi

# Step 2: Fine-tune the model
echo "=== Fine-tuning the model ==="
python training/fine_tune.py --config "$CONFIG_FILE" --data_dir "$DATA_DIR/processed" --output_dir "$OUTPUT_DIR" 2>&1 | tee training.log
if [ $? -ne 0 ]; then
    echo "Error: Fine-tuning failed!"
    exit 1
fi

echo "=== Training complete! ==="
echo "The fine-tuned model is saved in: $OUTPUT_DIR"