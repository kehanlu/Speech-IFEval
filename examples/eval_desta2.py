import argparse
import os
import json
import torch
from transformers import AutoModel
from pathlib import Path
import logging
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="DeSTA-ntu/DeSTA2-8B-beta")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    return parser.parse_args()

def get_response(args, data, model, tokenizer=None):
    instruction = data["instruction"]

    # TODO: Replace with actual model inference logic
    messages = [
                {"role": "system", "content": "Follow the given instructions."},
                {"role": "audio", "content": args.data_dir + "/audios/" + data["audio_filepath"]},
                {"role": "user", "content": f"{instruction}"},
            ]

    generated_ids = model.chat(messages, max_new_tokens=2048, do_sample=False, temperature=1, top_p=1.0)

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response, messages

def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.model_id.replace("/", "--")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    
    manifest_paths = [
        Path(data_dir) / "eval_data/closed_ended_questions.jsonl",
        Path(data_dir) / "eval_data/creative_writing.jsonl",
        Path(data_dir) / "eval_data/chain-of-thought.jsonl",
        Path(data_dir) / "eval_data/closed_ended_questions-woprompt.jsonl",
    ]

    # Load model
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=os.getenv("HF_HOME"), token=os.getenv("HF_TOKEN"))
    model.to("cuda")

    for manifest_path in manifest_paths:
        output_file = output_dir / manifest_path.name

        # logging to a file path that is the same as the manifest file
        logging.basicConfig(filename=output_dir / f"{manifest_path.stem}.log", level=logging.INFO)
        
        logging.info(f"Processing {manifest_path}")
        logging.info(f"Output file: {output_file}")

        with manifest_path.open("r") as fin, output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines()]


            for data in tqdm(datas):
                
                response, messages = get_response(args, data, model)
                
                data["messages"] = messages
                data["response"] = response  # Add response to data
                
                fout.write(json.dumps(data) + "\n")
                logging.info(json.dumps(data))

if __name__ == "__main__":
    args = arg_parser()
    main(args)