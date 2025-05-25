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
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.model_id.replace("/", "--")
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    
    manifest_paths = [
        Path(data_dir) / "eval_data/closed_ended_questions.jsonl",
        Path(data_dir) / "eval_data/creative_writing.jsonl",
        Path(data_dir) / "eval_data/chain-of-thought.jsonl",
        Path(data_dir) / "eval_data/closed_ended_questions-woprompt.jsonl",
    ]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=os.getenv("HF_HOME"),
        token=os.getenv("HF_TOKEN"),
    )

    for manifest_path in manifest_paths:
        output_file = output_dir / manifest_path.name

        # logging to a file path that is the same as the manifest file
        logging.basicConfig(filename=output_dir / f"{manifest_path.stem}.log", level=logging.INFO)
        
        logging.info(f"Processing {manifest_path}")
        logging.info(f"Output file: {output_file}")

        with manifest_path.open("r") as fin, output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines()]


            for data in tqdm(datas):
                instruction = data["instruction"]
                textual_audio = data["textual_audio"]

                # TODO: Replace with actual model inference logic
                content = f"""Speech Input: {textual_audio}\n\n{instruction}"""
                messages = [
                    {"role": "system", "content": "Follow the given instructions."},
                    {"role": "user", "content": content},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(model.device)

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                )
                
                response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
                data["messages"] = messages
                data["response"] = response  # Add response to data
                
                fout.write(json.dumps(data) + "\n")
                logging.info(json.dumps(data))

                break

if __name__ == "__main__":
    args = arg_parser()
    main(args)