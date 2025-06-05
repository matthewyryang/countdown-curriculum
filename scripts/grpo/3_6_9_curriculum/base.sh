#!/bin/bash
#SBATCH --job-name=ray-single
#SBATCH --partition=preempt # Or your desired partition
#SBATCH --nodes=1           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node (for ray start)
#SBATCH --gres=gpu:8        # 8 GPUs per node
#SBATCH --cpus-per-task=96  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=1024G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --qos=preempt_qos
#SBATCH --account=aviralku

source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

cd /home/myang4/countdown-curriculum

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
export EXPERIMENT_NAME="base"
export OUTPUT_DIR=/project/flame/myang4/$EXPERIMENT_NAME

# export EPOCHS=1
# export DATA_DISTRIBUTION="3-4-7"
# export CONTEXT_LENGTH=1024
# export BASE_MODEL=/project/flame/myang4/cognitive-behaviors-Llama-3.2-3B
# bash /home/myang4/countdown-curriculum/scripts/grpo/3_6_9_curriculum/grpo.sh > /home/myang4/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.log 2>&1

export EPOCHS=2
export DATA_DISTRIBUTION="3-4-7"
export CONTEXT_LENGTH=1024
bash /home/myang4/countdown-curriculum/scripts/convert_to_hf_model.sh $OUTPUT_DIR/global_step_30 > /home/myang4/countdown-curriculum/logs/$EXPERIMENT_NAME-part-1.5.log 2>&1
export BASE_MODEL=$OUTPUT_DIR/global_step_30/hf
bash /home/myang4/countdown-curriculum/scripts/grpo/3_6_9_curriculum/grpo.sh > /home/myang4/countdown-curriculum/logs/$EXPERIMENT_NAME-part-2.log 2>&1

