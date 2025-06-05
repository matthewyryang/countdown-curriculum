python /home/myang4/countdown-curriculum/scripts/model_merger.py --backend fsdp \
    --hf_model_path d1shs0ap/cognitive-behaviors-Llama-3.2-3B \
    --local_dir "$1/actor" \
    --target_dir "$1/hf"

python /home/myang4/countdown-curriculum/scripts/save_tokenizer.py \
    --hf_model_path d1shs0ap/cognitive-behaviors-Llama-3.2-3B \
    --target_dir "$1/hf"
