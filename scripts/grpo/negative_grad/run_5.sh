export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export GRAD="positive"
export DATA_DISTRIBUTION="train-5-5"
export CONTEXT_LENGTH=2048
export EXPERIMENT_NAME="$GRAD"


export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B
export EPOCHS=2
bash /home/cmu/countdown-curriculum/scripts/grpo/negative_grad/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_24
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_24/hf
export EPOCHS=4
bash /home/cmu/countdown-curriculum/scripts/grpo/negative_grad/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1


bash /home/cmu/countdown-curriculum/scripts/convert_to_hf_model.sh /home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_48
export BASE_MODEL=/home/cmu/countdown-curriculum/checkpoints/$EXPERIMENT_NAME/global_step_48/hf
export EPOCHS=6
bash /home/cmu/countdown-curriculum/scripts/grpo/negative_grad/grpo.sh > /home/cmu/countdown-curriculum/logs/$EXPERIMENT_NAME-part-3.log 2>&1

