import argparse, pickle, os, json
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


def apply_template(problem):
    messages = [
        {"role": "user", "content": problem}
    ]
    return messages

def construct_problem(entry):
    nums = entry['nums']
    assert len(nums) == 2

    return f"What is the product of {nums[0]} and {nums[1]}? Show your work in <think> </think> tags. Return the final answer in <answer> </answer> tags, for example <answer> 15234230 </answer>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", type=str, default='/data/user_data/myang4/multiply/multiply-train-2.json')
    # parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--dataset_start", type=int, default=None)
    parser.add_argument("--dataset_end", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="/data/user_data/myang4/multiply")
    parser.add_argument("-K", type=int, default=1)
    parser.add_argument("--model", type=str, default="/data/group_data/rl/myang4/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    with open(args.input_dataset_file, "r") as f:
        dataset = json.load(f)

    if args.dataset_start is None:
        args.dataset_start = 0
    if args.dataset_end is None:
        args.dataset_end = len(dataset)

    print(args.dataset_start, args.dataset_end, args.model)

    # dataset = load_dataset(args.input_dataset_name, split=args.dataset_split)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=20000)

    tokenizer = llm.get_tokenizer()

    solutions = {}
    for i in tqdm(range(args.dataset_start, args.dataset_end, args.batch_size)):
        batch = dataset[i : min(i + args.batch_size, args.dataset_end)]
        # convs = [apply_template(construct_problem(entry)) for entry in batch]
        convs = [apply_template(entry['prompt']) for entry in batch]
        completions = llm.chat(
            messages=convs,
            sampling_params=SamplingParams(
                n=args.K,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
        )
        for j, completion in enumerate(completions):
            solutions[i + j] = completion


    new_data = {}
    for i in range(args.dataset_start, args.dataset_end):
        texts = [output.text for output in solutions[i].outputs]
    
        new_data[i] = dataset[i]
        new_data[i]["qwen32b-samples"] = texts

    with open(os.path.join(args.output_path, f"multiply-train-{args.difficulty}-inferenced-verify.json"), "w") as f:
        json.dump(new_data, f, indent=4)

