import argparse
import os
import random

import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, set_seed
from kgw_watermarking.watermark_reliability_release.watermark_processor import WatermarkLogitsProcessor

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_prompts(args) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto", dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    min_length = args.prompt_length + args.min_new_tokens

    def filter_length(example):
        return len(tokenizer(example[args.data_field])['input_ids']) >= min_length

    def encode_text(examples):
        trunc_tokens = tokenizer(
            examples[args.data_field],
            truncation=True,
            padding=True,
            max_length=min_length,
            return_tensors="pt"
        ).to(device)
        examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
        prompt = tokenizer(
            examples["text"], truncation=True, padding=True, max_length=args.prompt_length, return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        examples["text_completion"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, args.prompt_length:], skip_special_tokens=True
        )
        return examples
    
    #! some issues with HF servers
    # dataset = load_dataset("json", split="train", data_files=f"final_watermarked_text/test/wikipedia_prompts.json")
    # dataset = dataset.map(encode_text, batched=True)
    # dataset = dataset.shuffle(seed=args.seed)

    #! this is specefic to booksum
    # dataset = dataset.remove_columns(['is_aggregate', 'source', 'chapter_path', 'summary_path', 'book_id', 'summary_id', 'content',  'chapter_length', 'summary_name', 'summary_url', 'summary_text', 'summary_analysis', 'summary_length', 'analysis_length']) 

    #! this is for a normal dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)
    dataset = dataset.filter(filter_length)
    dataset = dataset.map(encode_text, batched=True)
    dataset = dataset.shuffle(seed=args.seed)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    prompts = []
    human_text = []
    prompt_text = []
    full_human_text = []
    for batch in dataloader:
        if len(human_text) >= args.num_samples:
            break
        if (type(batch["input_ids"]) == list):
            batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
        if (type(batch["attention_mask"]) == list):
            batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
        prompts.append(batch)
        human_text.extend(batch["text_completion"])
        prompt_text.extend(batch["prompt_text"])
        full_human_text.extend(batch["text"])
    human_text = human_text[:args.num_samples]
    prompt_text = prompt_text[:args.num_samples]
    full_human_text = full_human_text[:args.num_samples]
    return {
        "prompts": prompts,
        "human_text": human_text,
        "prompt_text": prompt_text,
        "full_human_text": full_human_text,
    }

def generate_samples(model, tokenizer, args, prompts, watermark, watermark_config, do_sample) -> Dict:
    set_seed(args.seed)
    model_text = []
    full_model_text = []

    for batch in tqdm(prompts):
        if len(model_text) >= args.num_samples:
            break

        if watermark_config["type"] == "kth" and watermark.num_shifts > 1:
            watermark.cur_shift = random.choice(watermark.possible_shifts)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=True,
                max_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                logits_processor=LogitsProcessorList([watermark]),
                pad_token_id=tokenizer.eos_token_id,
            )

            n_input_tokens = batch["input_ids"].shape[1]
            model_text.extend(tokenizer.batch_decode(outputs[:, n_input_tokens:], skip_special_tokens=True))
            full_model_text.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # model_text discards the prompt, full_model_text contains the prompt
    model_text = model_text[:args.num_samples]
    full_model_text = full_model_text[:args.num_samples]
    samples = {"model_text": model_text, "full_model_text": full_model_text}
    return samples

def main(args):

    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            input_dict = json.load(f)
        for key in input_dict:
            if key in args and "model_name" not in key and "file" not in key:
                setattr(args, key, input_dict[key])
        samples_dict = input_dict["samples"]
    else:
        samples_dict = {}

    prompts_dict = get_prompts(args)
    prompts = prompts_dict["prompts"]
    human_text = prompts_dict["human_text"]
    prompt_text = prompts_dict["prompt_text"]
    full_human_text = prompts_dict["full_human_text"]

    with open(args.watermark_configs_file, "r") as f:
        watermark_configs_list = json.load(f)

    prefix_count = 0

    #! Now we do the actual work here
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto", dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.float16)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    for watermark_config in tqdm(watermark_configs_list):

        assert watermark_config["type"] == "kgw" # currently we are just working in this domain
        watermark_config['hash_key'] = args.seed

        watermark = WatermarkLogitsProcessor(
                    vocab=tokenizer.get_vocab().values(),
                    gamma=watermark_config["gamma"],
                    delta=watermark_config["delta"],
                    seeding_scheme=watermark_config['seeding_scheme'],
                    hash_key=args.seed,
                    device=device)
        do_sample = True
        
        # have to use this to store results        
        simplified_model_name = [s for s in args.model_name.split("/") if s][-1]
        prefix = f"scheme{watermark_config['seeding_scheme']}-gamma{watermark_config['gamma']}-delta{watermark_config['delta']}-key{args.seed}"
        simplified_model_name = f"{prefix}-{simplified_model_name}"

        print(f"Generating samples for model {simplified_model_name}")
        try:
            samples = generate_samples(model, tokenizer, args, prompts, watermark, watermark_config, do_sample)
        except Exception as e:
            print(f"Error generating samples for model {simplified_model_name}: {e}")
            continue

        samples["human_text"] = human_text
        samples["prompt_text"] = prompt_text
        samples["full_human_text"] = full_human_text
        full_watermark_config = {}
        try:
            for k, v in vars(watermark).items():
                if isinstance(v, (str, int, float, bool, list)):
                    full_watermark_config[k] = v
            if watermark_config["type"] == "kgw":
                full_watermark_config["type"] = watermark_config["type"]
                full_watermark_config["kgw_device"] = "cuda"
        except Exception as e:
            print(f"Error loading watermark config for model {simplified_model_name}: {e}")


        if full_watermark_config:
            samples["watermark_config"] = full_watermark_config
        elif watermark_config:
            samples["watermark_config"] = watermark_config
        samples["model_name"] = simplified_model_name
        samples_dict[simplified_model_name] = samples

        del watermark


    del model
    torch.cuda.empty_cache()

    output_dict = {
        "samples": samples_dict,
    }
    output_dict.update(vars(args))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        print(f"Writing output to {args.output_file}")
        json.dump(output_dict, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None, required=False)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_num_skip", type=int, default=0)
    parser.add_argument("--data_field", type=str, default="text")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--min_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--prompt_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--overwrite_output_file", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--watermark_configs_file", type=str, required=True)

    args = parser.parse_args()

    main(args)