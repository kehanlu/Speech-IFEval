
# Speech-IFEval

> **Speech-IFEval: Evaluating Instruction-Following and Quantifying Catastrophic Forgetting in Speech-Aware Language Models**
> 
> Ke-Han Lu, Chun-Yi Kuan and Hung-yi Lee
> National Taiwan University

We aim to evaluate the **textual knowledge** of Speech-Aware Language Models (SLMs).
- We find that most SLMs suffer from catastrophic forgetting after speech-text training.
- Existing benchmarks focus only on task-level performance (e.g., accuracy), making it unclear whether poor results stem from speech perception issues or difficulty understanding textual instructions or questions.

## 🏆 Leaderboard

| Rank | Model            | Closed-ended (%) | Creative Writing (%) | CoT (%) | **IFrate (%)** | **Δ (Forgetting Rate)** |
|------|------------------|------------------|-----------------------|---------|----------------|----------------------|
| 1    | DeSTA2        | 83.71             | 91.75                 | 91.50   | **89.23**       | **-3.57**               |
| 2    | DiVA              | 83.14             | 61.75                 | 83.50   | 76.13           | -17.73              |
| 3    | BLSP-emo          | 66.35             | 63.75                 | 50.50   | 60.20           | -17.92              |
| 4    | Qwen2-Audio-Instruct | 41.59         | 67.75                 | 32.00   | 47.11           | –                    |
| 5   | SALMONN           | 37.41             | 61.25                 | 12.00   | 36.89           | -50.20              |
| 6    | Qwen-Audio-Chat   | 10.93             | 56.00                 | 32.00   | 32.98           | –                    |
| 7    | LTU-AS            | 28.83             | 47.75                 | 11.00   | 29.19           | -54.90              |

> **Note:** IFrate is the average of Closed-ended, Creative Writing, and CoT following rates.  
> Forgetting Rate (Δ) is computed relative to each model’s original text-only LLM.
> 
> Qwen-audio series use Qwen-7B as their backbone, which is *not* instruction-tuned. Therefore, no reference system is available for Δ calculation.

> 📬 If you have evaluated your model using **Speech-IFEval**, feel free to send your results to us. Once verified, we will update the leaderboard to include your entry!

---

## 📊 Evaluate your model

### 🔧 Setup

```bash
git clone https://github.com/kehanlu/Speech-IFEval.git
cd Speech-IFEval
pip install -r requirements.txt
```

📥 Download Audio Files

```bash
cd data
wget https://huggingface.co/datasets/kehanlu/Speech-IFEval/resolve/main/audios.tar
tar -xvf audios.tar
```

**Directory structure:**

```
data/
│── eval_data/
│   │── closed_ended_questions.jsonl             # Closed-ended tasks
│   │── creative_writing.jsonl                   # Creative writing tasks
│   │── chain-of-thought.jsonl                   # CoT reasoning tasks
│   │── closed_ended_questions-woprompt.jsonl    # Baseline version of closed-ended tasks (optional)
│
│── audios/
│   │── Automatic_speech_recognition/
│   │── Gender_recognition/
│   │── Speech_emotion_recognition/
│   │── MMAU/
```


### 1. Evaluate Instruction-Following Rate (IFrate)

Run your Speech-aware Language Model (SLM) evaluation (e.g., **DeSTA2**):

```bash
python examples/eval_desta2.py --data /lab/Speech-IFEval/data --output_dir outputs
```

Then compute IFrate with:

```bash
# Closed-ended and Creative Writing evaluation
python -m instruction_following_eval.evaluation_main -i outputs/DeSTA-ntu--DeSTA2-8B-beta/closed_ended_questions.jsonl
python -m instruction_following_eval.evaluation_main -i outputs/DeSTA-ntu--DeSTA2-8B-beta/creative_writing.jsonl

# Chain-of-Thought (CoT) reasoning evaluation
python script/llm_evaluation.py -i outputs/DeSTA-ntu--DeSTA2-8B-beta/chain-of-thought.jsonl --stage 0
```

**Example Results (DeSTA2):**

| Task              | Following Rate |
|------------------|----------------|
| Closed-ended      | 83.71%         |
| Creative Writing  | 91.75%         |
| Chain-of-Thought  | 91.50%         |
| **IFrate**        | **89.23%**     |

---

### 2. Evaluate Forgetting Rate (Δ)

> With a reference system, we can assess the forgetting rate by comparing the speech-aware model to its text-only counterpart, thereby quantifying the degradation introduced by speech-text training.


Run the reference system baseline (e.g., **Llama3-8B-Instruct for DeSTA2**):

```bash
python examples/eval_llama3_8B_instruct.py --data /lab/Speech-IFEval/data --output_dir outputs
```

**Reference System Results:**

| Task              | Following Rate |
|------------------|----------------|
| Closed-ended      | 93.35%         |
| Creative Writing  | 93.75%         |
| Chain-of-Thought  | 90.50%         |
| **IFrate**        | **92.53%**     |


**Calculate Forgetting Rate (Δ)**

$$
Δ = (IFrate_{SLM} - IFrate_{Ref}) / (IFrate_{Ref}) = (89.23 - 92.53) / (92.53) = -3.57
$$

| Model            | IFrate | Δ (Forgetting Rate) |
|------------------|--------|---------------------|
| Llama3-8B-Instruct | 92.53% | --                  |
| DeSTA2            | 89.23% | -3.57%              |

---

### 📌 (Optional) Task-Level Evaluation

To replicate **Table 4** from the paper (with and without output constraints):

```bash
# Without constraint prompt (baseline task-level performance)
python script/llm_evaluation.py -i outputs/DeSTA-ntu--DeSTA2-8B-beta/closed_ended_questions-woprompt.jsonl --stage 0

# With constraint prompt
python script/llm_evaluation.py -i outputs/DeSTA-ntu--DeSTA2-8B-beta/closed_ended_questions.jsonl --stage 0
```


### Citation

```
@inproceedings{Lu_interspeech,
  title     = {Speech-IFEval: Evaluating Instruction-Following and Quantifying Catastrophic Forgetting in Speech-Aware Language Models},
  author    = {Ke-Han Lu, Chun-Yi Kuan and Hung-yi Lee},
  year      = {2025},
  booktitle = {Interspeech 2025},
}
```
