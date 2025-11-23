#!/bin/bash

# Initialize MODEL_PATH as an empty string
MODEL_PATH=""

# Parse arguments
while [ $# -gt 0 ]; do
  case $1 in
    *=*)
      # Split the argument into key and value
      key=$(echo "$1" | cut -d '=' -f 1)
      value=$(echo "$1" | cut -d '=' -f 2)
      
      if [ "$key" == "MODEL_PATH" ]; then
        MODEL_PATH="$value"
      fi

      # Dynamically set the environment variable
      export "$key=$value"
      ;;
    *)
      # If it's not in key=value format, treat it as MODEL_PATH (if not already set)
      if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH=$1
      else
        echo "Unexpected argument: $1"
        exit 1
      fi
      ;;
  esac
  shift  # Move to the next argument
done

# Check if MODEL_PATH is provided
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required."
  exit 1
fi

# Set default values
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-cuda:0}
MODEL=${MODEL:-resnet18}
TRANSFORM=${TRANSFORM:-224}

python ../sl_train/test_ae_sl.py  \
    --batch_size  "$BATCH_SIZE"     \
    --device      "$DEVICE"         \
    --model       "$MODEL"          \
    --model_path  "$MODEL_PATH"     \
    --transform   "$TRANSFORM"      
    