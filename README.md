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

manifest_paths = [
    Path("./data/close-open.jsonl"),
    Path("./data/close-woprompt.jsonl"),
    Path("./data/chain-of-thought.jsonl"),
]
output_dir = Path("/path/to/outputs/")
output_dir.mkdir(parents=True, exist_ok=True)

for manifest_path in manifest_paths:
    output_file = output_dir / manifest_path.name

    with manifest_path.open("r") as fin, output_file.open("w") as fout:
        for line in fin:
            data = json.loads(line)

            audio_filepath = data.["audio_filepath"]
            instruction = data.["instruction"]

            # TODO: Replace with actual model inference logic
            response = f"Generated response for {audio_filepath}"
            
            
            data["response"] = response  # Add response to data
            fout.write(json.dumps(data) + "\n")

```


Evaluate instruction following:

```
python3 -m instruction_following_eval.evaluation_main --input_response_data=/path/to/outputs/close-open.jsonl
```
