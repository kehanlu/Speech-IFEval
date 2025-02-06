import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import logging
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-7B-Chat")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()

def main(args):

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.model_id.replace("/", "--")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    manifest_paths = [
        Path(data_dir) / "eval_data/close.jsonl",
        Path(data_dir) / "eval_data/open.jsonl",
        Path(data_dir) / "eval_data/close-woprompt.jsonl",
        Path(data_dir) / "eval_data/chain-of-thought.jsonl",
    ]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map=args.device,
        cache_dir=os.getenv("HF_HOME"),
        token=os.getenv("HF_TOKEN"),).eval()
        
    for manifest_path in manifest_paths:
        output_file = output_dir / manifest_path.name

        # logging to a file path that is the same as the manifest file
        logging.basicConfig(filename=output_dir / f"{manifest_path.stem}.log", level=logging.INFO)
        
        logging.info(f"Processing {manifest_path}")
        logging.info(f"Output file: {output_file}")


        with manifest_path.open("r") as fin, output_file.open("w") as fout:
            for line in tqdm(fin):
                data = json.loads(line)

                audio_filepath = data["audio_filepath"]
                instruction = data["instruction"]
                seed_transcript = data["seed_transcript"]


                # TODO: Replace with actual model inference logic
                content = f"""Speech Input: {seed_transcript}\n\n{instruction}"""
                
                response, history = model.chat(tokenizer, content, history=None)

                data["response"] = response  # Add response to data
                
                fout.write(json.dumps(data) + "\n")
                logging.info(json.dumps(data))

if __name__ == "__main__":
    args = arg_parser()
    main(args)