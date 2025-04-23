LOCAL_DIR=/home/cmu/countdown-curriculum/checkpoints/3-4-256-both-curriculum/global_step_150/actor
TARGET_DIR=/home/cmu/countdown-curriculum/checkpoints/3-4-256-both-curriculum/global_step_150/hf

python scripts/model_merger.py --backend fsdp \
    --hf_model_path d1shs0ap/cognitive-behaviors-Llama-3.2-3B \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR

python scripts/save_tokenizer.py \
    --hf_model_path d1shs0ap/cognitive-behaviors-Llama-3.2-3B \
    --target_dir $TARGET_DIR
