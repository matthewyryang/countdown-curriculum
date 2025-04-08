export CUDA_VISIBLE_DEVICES=4,5

CONTEXT_LENGTH=1024
DATA_DISTRIBUTION="countdown-3-4-5-6"

export N_GPUS=2
export BATCH_SIZE_PER_GPU=8
export CONTEXT_LENGTH=$CONTEXT_LENGTH

export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B

export DATA_DIR="/home/cmu/countdown-curriculum/data/$DATA_DISTRIBUTION"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-no-positive-grad"
export OUTPUT_DIR="/data/user_data/cmu/countdown/$EXPERIMENT_NAME"
export SAVE_FREQ=-1

# ray stop --force && ray start --head

nohup bash /home/cmu/countdown-curriculum/scripts/grpo/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
