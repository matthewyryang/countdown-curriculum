export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export GRAD="positive"
export PREV_EXPERIMENT_NAME="hard-length-curriculum"
export EXPERIMENT_NAME="$PREV_EXPERIMENT_NAME-$GRAD"
export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export DATA_DISTRIBUTION="1-easy-2-hard"


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$PREV_EXPERIMENT_NAME/global_step_75
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$PREV_EXPERIMENT_NAME/global_step_75/hf
export CONTEXT_LENGTH=1024
export EPOCHS=1
bash /home/cmu/countdown-curriculum/scripts/grpo/negative_grad/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1

