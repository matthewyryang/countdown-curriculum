export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export EXPERIMENT_NAME="both-curriculum"
export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B



export DATA_DISTRIBUTION="1-easy"
export CONTEXT_LENGTH=256
export EPOCHS=1
bash /home/cmu/countdown-curriculum/scripts/grpo/both_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_25
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_25/hf
export DATA_DISTRIBUTION="1-easy-2-medium"
export CONTEXT_LENGTH=512
export EPOCHS=1
bash /home/cmu/countdown-curriculum/scripts/grpo/both_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_100
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_100/hf
export DATA_DISTRIBUTION="1-easy-1-medium-3-hard"
export CONTEXT_LENGTH=1024
export EPOCHS=1
bash /home/cmu/countdown-curriculum/scripts/grpo/both_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-3.log 2>&1
