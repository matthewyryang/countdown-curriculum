export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export EXPERIMENT_NAME="difficulty"
export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export CONTEXT_LENGTH=1024


export DATA_DISTRIBUTION="train-3-3"
export EPOCHS=5
bash /home/cmu/countdown-curriculum/scripts/grpo/3_6_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60/hf
export DATA_DISTRIBUTION="train-6-6"
export EPOCHS=10
bash /home/cmu/countdown-curriculum/scripts/grpo/3_6_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1

