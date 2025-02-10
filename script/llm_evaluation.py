import argparse
import os
import json
import torch
from pathlib import Path
import logging
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer
from collections import defaultdict
from jiwer import wer
import re
from openai import OpenAI
from time import sleep
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def normalize_text(text):
    normalizer = BasicTextNormalizer()
    return normalizer(text).strip()

def extract_result(text):
    pattern = r"(?i)(?<=result:\s)(yes|no)"
    match = re.search(pattern, text)
    if match:
        return match.group(0)  # Convert to lowercase
    return None



def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_response_data", "-i", type=str, required=True)
    parser.add_argument("--stage", type=int, default=0)

    return parser.parse_args()

def generate_eval_response(data):
    instruction = data["instruction"]
    label = data.get("label")
    model_response = data["response"]

    if data["metric"] == "accuracy":
        assert label
        system_prompt = f"""You will be given a question, a corresponding correct answer and a response from a model. 
Model's Response is a reply to the Question. Your task is to judge if "Model's Response" aligns with the "Ground Truth Answer" based on the "Question". 
Please strictly follow the guidelines below:
- Answer with the format "Result: <YES or NO>" at the end.
- Output "YES" if the response aligns with the ground truth answer; output "NO" if the response does not match the ground truth answer.
"""
        content = f"""Question: {instruction}\nGround Truth Answer: {label}\nModel's Response: {model_response}"""

    elif data["metric"] == "wer":
        system_prompt = f"""You will be given a response from an ASR model. Your task is to extract a **substring** from the model's response that eliminates all extra phrases, explanations, or introductory text. The substring will be evaluate by the WER metric, so it should be **exactly the same** as the model's response, with no modifications.\n\nPlease strictly follow the guidelines below:\n- The substring should be **exactly the same** as the model's response, with no modifications.\n- Eliminate all extra phrases, explanations, or introductory text while keeping the substring itself 100% unchanged.\n- You must output the substring only."""
        content = f"""Question: {instruction}\nModel's Response: {model_response}"""

    elif data["metric"] == "cot":
        system_prompt = f"""You will be given a user input and a model response. The model's response is a reply to the user input. Your task is to determine whether the response demonstrates reasoning behavior, such as breaking down the problem, explaining intermediate steps, or providing a analysis.

Please strictly follow the guidelines below:
- Output "YES" if the response includes any form of behavior beyond a direct answer corresponding to the user input.
- Output "NO" only if the response is a minimal or purely factual reply.
- Answer in the format: "Result: <YES or NO>" at the end.
"""
        content = f"""User input: {instruction}\nModel's Response: {model_response}"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": content
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
    )
    response = response.choices[0].message.content
    if data["metric"] == "wer" and (normalize_text(response) not in normalize_text(model_response)):
        logging.warning(f"{'='*79}\n{normalize_text(model_response)}\n{'*'*79}\n{normalize_text(response)}\n{'='*79}")
    
    sleep(0.3)
    return response

def main(args):
    input_response_data_path = Path(args.input_response_data)

    output_dir = input_response_data_path.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output_dir / "logs" / f"{input_response_data_path.stem}.log", level=logging.INFO)

    tmp_output_file = output_dir / "tmp" / f"{input_response_data_path.stem}.jsonl"
    
    if args.stage < 1:
        logging.info(f"=================== LLM evaluation ====================")
        logging.info(f"Processing {input_response_data_path}")
        logging.info(f"Output file: {tmp_output_file}")
        with input_response_data_path.open("r") as fin, tmp_output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines()]
            for data in tqdm(datas):
                if data["metric"] != "accuracy":
                    continue
                response = generate_eval_response(data)
                data["eval_response"] = response
                fout.write(json.dumps(data) + "\n")
                logging.info(json.dumps(data))

    output_file = output_dir / f"llm_eval@{input_response_data_path.stem}.jsonl"
    if args.stage < 2:
        logging.info(f"=================== Performance Evaluation ====================")
        with tmp_output_file.open("r") as fin, output_file.open("w") as fout:
            datas = [json.loads(line) for line in fin.readlines()]
            
            dataset_group = defaultdict(list)
            hyp = []
            ref = []
            for data in tqdm(datas):
                if data["metric"] == "accuracy":
                    result = extract_result(data["eval_response"])
                    if result.lower() == "yes":
                        dataset_group[data["dataset"]].append(1)
                        data["correct"] = True
                    else:
                        dataset_group[data["dataset"]].append(0)
                        data["correct"] = False
                
                elif data["metric"] == "wer":
                    hyp.append(normalize_text(data["eval_response"]))
                    ref.append(normalize_text(data["label"]))
                    data["correct"] = wer(truth=ref, hypothesis=hyp)

                elif data["metric"] == "cot":
                    result = extract_result(data["eval_response"])
                    if result.lower() == "yes":
                        dataset_group["cot"].append(1)
                        data["correct"] = True
                    else:
                        dataset_group["cot"].append(0)
                        data["correct"] = False

                fout.write(json.dumps(data) + "\n")
        
    # print report
    if ref:
        wer_score = wer(truth=ref, hypothesis=hyp)
        logging.info(f"WER: {wer_score}")
        print(f"WER: {wer_score}")
    for dataset, correct in dataset_group.items():
        logging.info(f"{dataset} ACC: {sum(correct)/len(correct)}")
        print(f"{dataset} ACC: {sum(correct)/len(correct)}")

if __name__ == "__main__":
    args = arg_parser()
    main(args)