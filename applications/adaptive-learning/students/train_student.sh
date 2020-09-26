#!/bin/bash

set -e

helpFunction()
{
   echo ""
   echo "Usage: $0 -c [An adaptive learning config file]"
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
echo "====================================================="
echo "Adaptive Learning Process Started..."
echo "====================================================="
sleep 5
Quantization_Aware=$(sed -n -e 's/^\s*QuantizationAware\s*:\s*//p' "$CONFIG_FILE")
if $Quantization_Aware
then
  TRAINING_PIPELINE_FILE="/repo/applications/adaptive-learning/students/ssd_mobilenet_v2_pedestrian_quant.config"
else
  TRAINING_PIPELINE_FILE="/repo/applications/adaptive-learning/students/ssd_mobilenet_v2_pedestrian.config"
fi
	
python apply_configs.py --config "$CONFIG_FILE" --pipeline $TRAINING_PIPELINE_FILE

# extrat config items
TRAINING_ROUNDS=$(sed -n -e 's/^\s*TrainingRounds\s*:\s*//p' "$CONFIG_FILE")
DATASET_DIR=$(sed -n -e 's/^\s*DataDir\s*:\s*//p' "$CONFIG_FILE")
BATCHSIZE=$(sed -n -e 's/^\s*BatchSize\s*:\s*//p' "$CONFIG_FILE")
EPOCHS=$(sed -n -e 's/^\s*Epochs\s*:\s*//p' "$CONFIG_FILE")
EXAMPLE_PER_ROUND=$(sed -n -e 's/^\s*ExamplePerRound\s*:\s*//p' "$CONFIG_FILE")
Validation_Split=$(sed -n -e 's/^\s*ValidationSplit\s*:\s*//p' "$CONFIG_FILE")

NUM_TRAIN_STEPS=$(( EXAMPLE_PER_ROUND * EPOCHS / BATCHSIZE ))

# download pretrained checkpoint if it does not exist
PRETRAINED_MODELS_DIR="/repo/applications/adaptive-learning/data/pretrained_models"
mkdir -p $PRETRAINED_MODELS_DIR
if [ ! -f $PRETRAINED_MODELS_DIR/ssd_mobilenet_v2_coco_2018_03_29.tar.gz ]
then
  echo "The Pretrained checkpoints are not exists"
  echo "Start Downloading Pretrained Checkpoits"
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz \
  -O $PRETRAINED_MODELS_DIR/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  tar -xzvf $PRETRAINED_MODELS_DIR/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C $PRETRAINED_MODELS_DIR
fi


MODEL_DIR="/repo/applications/adaptive-learning/data/student_model"
EXPORT_DIR=$MODEL_DIR/frozen_graph
TFRECORD_DIR="/repo/applications/adaptive-learning/data/tfrecords"
mkdir -p $MODEL_DIR
mkdir -p $EXPORT_DIR
mkdir -p $TFRECORD_DIR
# TODO: add infinite loop option
for ((i=1;i<=TRAINING_ROUNDS;i++))
do
  echo "============Round $i of Training Started================="

  echo "=================================================================================="
  echo "Start Creating TFrecords ..."
  echo "=================================================================================="
  python create_tfrecord.py --data_dir "$DATASET_DIR" \
  --output_dir $TFRECORD_DIR \
  --label_map_path "/repo/applications/adaptive-learning/students/label_map.pbtxt" \
  --validation_split $Validation_Split \
  --num_of_images_per_round "$EXAMPLE_PER_ROUND"
  echo "=================================================================================="
  echo "Start Training ..."
  echo "=================================================================================="
  NUM_TRAIN_STEPS_NEW=$(( NUM_TRAIN_STEPS * i ))
  python /models/research/object_detection/model_main.py --pipeline_config_path=$TRAINING_PIPELINE_FILE \
  --model_dir=$MODEL_DIR \
  --num_train_steps=$NUM_TRAIN_STEPS_NEW \
  --sample_1_of_n_eval_examples=1 \
  --alsologtostderr 2>&1 | tee training_log.txt
  CHECKPOINT=$(ls -t $MODEL_DIR|grep ckpt|head -n 1 | tr -dc '0-9')
  TRAINED_CKPT_PREFIX=$MODEL_DIR/model.ckpt-$CHECKPOINT

  echo "=================================================================================="
  echo "Start Exporting Checkpoint to Frozen Graph ..."
  echo "=================================================================================="
  rm -rf $EXPORT_DIR/*
  if $Quantization_Aware
  then
    python /models/research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path=$TRAINING_PIPELINE_FILE \
    --trained_checkpoint_prefix="$TRAINED_CKPT_PREFIX" \
    --output_directory=$EXPORT_DIR \
    --max_detections=50 \
    --add_postprocessing_op=true
  else
    python /models/research/object_detection/export_inference_graph.py --input_type=image_tensor \
    --pipeline_config_path=$TRAINING_PIPELINE_FILE \
    --trained_checkpoint_prefix="$TRAINED_CKPT_PREFIX" \
    --output_directory=$EXPORT_DIR 
  fi
done
