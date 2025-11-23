 #!/bin/bash

# Parse additional parameters (key=value pairs)
while [ $# -gt 0 ]; do
  case $1 in
    *=*)
      # Split the argument into key and value
      key=$(echo "$1" | cut -d '=' -f 1)
      value=$(echo "$1" | cut -d '=' -f 2)
      # Dynamically set the environment variable
      export "$key=$value"
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
      ;;
  esac
  shift  # Move to the next argument
done

# Set default values
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-100}
DEVICE=${DEVICE:-cuda:0}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MODEL=${MODEL:-resnet18}
ADV_SAMPLE_RATE=${ADV_SAMPLE_RATE:-0.5}
START_ADV_EPOCH=${START_ADV_EPOCH:-70}

python ../sl_train/train_sl.py            \
    --batch_size       "$BATCH_SIZE"        \
    --num_epochs       "$NUM_EPOCHS"        \
    --device           "$DEVICE"            \
    --learning_rate    "$LEARNING_RATE"     \
    --model            "$MODEL"             \
    --adv_sample_rate  "$ADV_SAMPLE_RATE"   \
    --start_adv_epoch  "$START_ADV_EPOCH"   
    
    