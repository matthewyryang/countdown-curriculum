export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export EXPERIMENT_NAME="6-no-curriculum"
export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export DATA_DISTRIBUTION="train-6-6"


export CONTEXT_LENGTH=512
export EPOCHS=5
bash /home/cmu/countdown-curriculum/scripts/grpo/5_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_60/hf
export CONTEXT_LENGTH=512
export EPOCHS=10
bash /home/cmu/countdown-curriculum/scripts/grpo/5_curriculum/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1

