export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export EXPERIMENT_NAME="length"
export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export DATA_DISTRIBUTION="3-and-6-x2.5"


# export CONTEXT_LENGTH=256
# export EPOCHS=1
# bash /home/cmu/countdown-curriculum/scripts/grpo/3_6_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1


# bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60
# export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60/hf
export CONTEXT_LENGTH=1024
export EPOCHS=2
bash /home/cmu/countdown-curriculum/scripts/grpo/3_6_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1

