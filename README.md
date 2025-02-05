## Audio data

```
wget https://huggingface.co/datasets/kehanlu/IFeval-Speech/resolve/main/audios.tar

tar -xvf audios.tar
```

**Folde structure:**
```
audios/
├── Automatic_speech_recognition/
├── Gender_recognition/
├── Speech_emotion_recognition/
└── MMAU/
```


## Evaluation


Input example:
```
{
  "audio_filepath": "Automatic_speech_recognition/1995-1837-0019.flac",
  "seed_transcript": "[00:00:00 - 00:00:05] He sat down, weak, bewildered, and one thought was uppermost. Zora! Zora!(Gender: Female, Emotion: sad)",
  "instruction": "Convert the provided spoken statement into text.\nYour entire response should be in all capital letters.",
  "dataset": "Automatic_speech_recognition",
  "metric": "wer",
  "instruction_id_list": [
    "change_case:english_capital"
  ],
  "kwargs": [
    {}
  ],
  "id": 1
}
```

Infrence your model:

```python
import json
from pathlib import Path

manifest_paths = ["./data/close-open.jsonl", "./data/close-woprompt.jsonl", "./data/cot.jsonl"]
output_dir = "/path/to/outputs/"

for manifest_path in manifest_paths:
    manifest_path = Path(manifest_path)
    
    with open(Path(output_dir) / manifest_path.name, "w") as fout:
        audio_filepath = data["audio_filepath"]
        instruction = data["instruction"]

        for line in manifest_path.open().readlines():
            data = json.loads(line)

            # TODO: code for inference your model
            response = model(data)

            data["response"] = response # add a item into data then write outputs

            fout.write(json.dumps(data) + "\n")
```


Evaluate instruction following:

```
python3 -m instruction_following_eval.evaluation_main --input_response_data=/path/to/outputs/close-open.jsonl
```
