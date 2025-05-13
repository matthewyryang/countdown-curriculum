import pandas as pd
import argparse, os, pickle
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    llm = LLM(model="/home/cmu/countdown-curriculum/checkpoints/normal/global_step_72/hf")

    df = pd.read_parquet('/home/cmu/countdown-curriculum/data/countdown/test-5-5.parquet')
    prompts = []
    for i, row in df.iterrows():
        prompts.append(row['prompt'][0]['content'])
    
    subset = prompts[args.gpu * len(prompts) // 8 : (args.gpu + 1) * len(prompts) // 8]

    completions = llm.generate(prompts=prompts, sampling_params=SamplingParams(n=16, temperature=0.6, top_p=0.95, max_tokens=2048))

    rollouts_file = f"/home/cmu/countdown-curriculum/checkpoints/normal/72_rollouts_16_gpu_{args.gpu}.pkl"
    with open(rollouts_file, 'wb') as f:
        pickle.dump(completions, f)
