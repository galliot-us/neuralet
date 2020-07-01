#!/bin/bash

set -e

helpFunction()
{
   echo ""
   echo "Usage: $0 -c [A bootstrap config file]"
   exit 1 # Exit script after printing help
}

while getopts "c:h:" opt
do
   # shellcheck disable=SC2220
   case "$opt" in

      c ) CONFIG_FILE="$OPTARG" ;;
      h ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$CONFIG_FILE" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
TRAINING_PIPELINE_FILE="/repo/applications/smart-distancing/bootstrap/students/ssd_mobilenet_v2_pedestrian.config"
python apply_configs.py --config "$CONFIG_FILE" --pipeline $TRAINING_PIPELINE_FILE

TRAINING_ROUNDS=$(sed -n -e 's/^\s*TrainingRounds\s*:\s*//p' "$CONFIG_FILE")
DATASET_DIR=$(sed -n -e 's/^\s*DataDir\s*:\s*//p' "$CONFIG_FILE")
BATCHSIZE=$(sed -n -e 's/^\s*BatchSize\s*:\s*//p' "$CONFIG_FILE")
EPOCHS=$(sed -n -e 's/^\s*Epochs\s*:\s*//p' "$CONFIG_FILE")
EXAMPLE_PER_ROUND=$(sed -n -e 's/^\s*ExamplePerRound\s*:\s*//p' "$CONFIG_FILE")

NUM_TRAIN_STEPS=$(( EXAMPLE_PER_ROUND * EPOCHS / BATCHSIZE ))


PRETRAINED_MODELS_DIR="/repo/applications/smart-distancing/bootstrap/data/pretrained_models"
mkdir -p $PRETRAINED_MODELS_DIR
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz \
-O $PRETRAINED_MODELS_DIR/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xzvf $PRETRAINED_MODELS_DIR/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C $PRETRAINED_MODELS_DIR


MODEL_DIR="/repo/applications/smart-distancing/bootstrap/data/student_model"
EXPORT_DIR=$MODEL_DIR/frozen_graph
TFRECORD_DIR="/repo/applications/smart-distancing/bootstrap/data/tfrecords"
mkdir -p $MODEL_DIR
mkdir -p $EXPORT_DIR
mkdir -p $TFRECORD_DIR

for ((i=1;i<=TRAINING_ROUNDS;i++))
do
  echo "============Round $i of Training Started================="
  python create_tfrecord.py --data_dir "$DATASET_DIR" \
  --output_dir $TFRECORD_DIR \
  --label_map_path "/repo/applications/smart-distancing/bootstrap/students/label_map.pbtxt" \
  --validation_split 0.10 \
  --num_of_images_per_round "$EXAMPLE_PER_ROUND"
  echo "=================================================================================="
  echo "=================================================================================="
  echo "=================================================================================="

  NUM_TRAIN_STEPS_NEW=$(( NUM_TRAIN_STEPS * i ))
  python /models/research/object_detection/model_main.py --pipeline_config_path=$TRAINING_PIPELINE_FILE \
  --model_dir=$MODEL_DIR \
  --num_train_steps=$NUM_TRAIN_STEPS_NEW \
  --sample_1_of_n_eval_examples=1 \
  --eval_training_data=True \
  --sample_1_of_n_eval_on_train_examples=9 \ 
  --alsologtostderr 2>&1 | tee log.txt
  CHECKPOINT=$(ls -t $MODEL_DIR|grep ckpt|head -n 1 | tr -dc '0-9')
  TRAINED_CKPT_PREFIX=$MODEL_DIR/model.ckpt-$CHECKPOINT

  rm -rf $EXPORT_DIR/*
  python /models/research/object_detection/export_inference_graph.py --input_type=image_tensor \
  --pipeline_config_path=$TRAINING_PIPELINE_FILE \
  --trained_checkpoint_prefix="$TRAINED_CKPT_PREFIX" \
  --output_directory=$EXPORT_DIR
done
