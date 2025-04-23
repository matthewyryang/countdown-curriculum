export CUDA_VISIBLE_DEVICES=4,5,6,7
export N_GPUS=4

export CONTEXT_LENGTH=512
export DATA_DISTRIBUTION="5-6"
export EPOCHS=3
# export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/3-4-256-both-curriculum/global_step_150/hf

export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-both-curriculum"

# ray stop --force && ray start --head

nohup bash /home/cmu/countdown-curriculum/scripts/grpo/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
