# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
import json
import os
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging

from instruction_following_eval import instructions_registry
from pathlib import Path


_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=False
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=False,
)


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def read_key_to_prompt_dict(input_jsonl_filename):
  """Creates dictionary matching key to prompt."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      return_dict[example["key"]] = InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"])
  return return_dict


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      # f.write(
      #     json.dumps(
      #         {
      #             attr_name: o.__getattribute__(attr_name)
      #             for attr_name in [
      #                 name for name in dir(o) if not name.startswith("_")
      #             ]
      #         }, ensure_ascii=False,
      #     )
      # )
      f.write(json.dumps(o, ensure_ascii=False))
      f.write("\n")


def test_instruction_following_strict(
    inp,
    result
):
  """Tests response to see if instrutions are followed."""
  response = result["response"]

  instruction_list = inp["instruction_id_list"]
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp["kwargs"][index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp["prompt"])

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  result["follow_instruction_list"] = is_following_list
  result["follow_all_instructions"] = all(is_following_list)
  return result

  # return OutputExample(
  #     instruction_id_list=inp.instruction_id_list,
  #     prompt=inp.prompt,
  #     response=response,
  #     follow_all_instructions=all(is_following_list),
  #     follow_instruction_list=is_following_list,
  # )


def test_instruction_following_loose(
    inp,
    response,
):
  """Tests response for an upper bound for following instructions."""
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )


def read_result_list(input_jsonl_filename):
  """Creates dictionary matching prompt and response."""
  results = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      results.append(example)
  return results


def print_report(outputs):
  """Prints a report on accuracy scores."""

  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  group_map = {
    "detectable_format:number_bullet_lists": "bullet_lists",
    "length_constraints:number_words": "length_constraints",
    "length_constraints:number_sentences": "length_constraints",
    "length_constraints:number_paragraphs": "length_constraints",
    "keywords:forbidden_words": "keywords",
    "keywords:existence": "keywords",
    "change_case:english_capital": "change_case",
    "change_case:english_lowercase": "change_case",
    "detectable_format:json_format": "json_format",
    "startend:quotation": "wrapping",
    "detectable_format:title": "wrapping",
    "combination:repeat_prompt": "startend",
    "startend:end_checker": "startend",
  }
  group_total = collections.defaultdict(int)
  group_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example["follow_instruction_list"]
    instruction_id_list = example["instruction_id_list"]

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      instruction_id = instruction_id.split(":")[0]
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      tier1_total[instruction_id] += 1
      if followed_or_not:
        tier1_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(
        instruction_id_list, follow_instruction_list
    ):
      group = group_map.get(instruction_id, "other")
      group_total[group] += 1
      if followed_or_not:
        group_correct[group] += 1

  print(f"prompt-level: {prompt_correct / prompt_total}")
  print(f"instruction-level: {instruction_correct / instruction_total}")
  print()
  for instruction_id in sorted(tier0_total.keys()):
    accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
    print(f"{instruction_id} {accuracy}")
  print()
  for instruction_id in sorted(tier1_total.keys()):
    accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
    print(f"{instruction_id} {accuracy}")

  print()
  print("===== Group accuracy =====")
  for group in sorted(group_total.keys()):
    accuracy = group_correct[group] / group_total[group]
    print(f"{group} {accuracy}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # inputs = read_key_to_prompt_dict(_INPUT_DATA.value)
  results = read_result_list(
      _INPUT_RESPONSE_DATA.value)
  print(len(results))

  # get instruction following results
  for func in [
      test_instruction_following_strict,
  ]:
    input_file_name = _INPUT_RESPONSE_DATA.value.split("/")[-1]
    output_file_name = f"rule_eval@{input_file_name}"
    logging.info("Generating %s...", output_file_name)
    outputs = []

    for result in results:
      condition = {
        "key": result["id"],
        "prompt": result["instruction"],
        "instruction_id_list": result["instruction_id_list"],
        "kwargs": result["kwargs"],
      }
      # InputExample(key=response["id"],
      #                  instruction_id_list=response["instruction_id_list"],
      #                  prompt=response["instruction"],
      #                  kwargs=response["kwargs"])

      outputs.append(func(condition, result))
    print(len(outputs))

    # for inp in inputs:
    #   outputs.append(func(inp, key_to_response))
    follow_all_instructions = [o["follow_all_instructions"] for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)

    (Path(_INPUT_RESPONSE_DATA.value).parent / "reports").mkdir(parents=True, exist_ok=True)

    output_file_name = str((Path(_INPUT_RESPONSE_DATA.value).parent / "reports") / f"{output_file_name}.jsonl")
    write_outputs(output_file_name, outputs)
    logging.info("Generated: %s", output_file_name)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    print_report(outputs)


if __name__ == "__main__":
  app.run(main)
