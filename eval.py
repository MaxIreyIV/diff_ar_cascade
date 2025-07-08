import re
import sys
from datasets import load_dataset
# from test import cascade_dream_to_deepseek
import time
from models import run_deepseek, run_dream

def get_question(example):
    return example["question"].strip()

def get_answer(example):
    return str(example["answer"]).strip()

def extract_number(text: str):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def extract_predicted_number(text: str):
    if "answer:" in text:
        text = text.split("answer:", 1)[1]

    raw_nums = re.findall(r"-?[0-9][0-9,]*", text)
    if not raw_nums:
        return None

    nums = [int(n.replace(",", "")) for n in raw_nums]
    return nums[-1]


def gold_answer(ans_str: str):
    if "####" in ans_str:
        ans_str = ans_str.split("####")[-1]
    return extract_number(ans_str)


def main(n_examples: int | None = None):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print("Dataset length:", len(ds))

    if n_examples is None or n_examples > len(ds):
        n_examples = len(ds)

    ds = ds.select(range(n_examples))
    # NUM_FEW_SHOT = 8
    # train_ds = load_dataset("openai/gsm8k", "main", split="train").select(range(NUM_FEW_SHOT))
    # few_shot_prefix = ""
    # for ex in train_ds:
    #     fs_q = get_question(ex)
    #     fs_a = get_answer(ex)
    #     few_shot_prefix += f"Question: {fs_q}\nAnswer: {fs_a}\n\n"

    correct = 0
    total_time = 0.0
    results = []
    for idx, example in enumerate(ds):
        q = get_question(example)
        gold = gold_answer(get_answer(example))

        # prompt = f"{few_shot_prefix}Question: {q.strip()}\nAnswer:"
        prompt = f"Question: {q.strip()}\nAnswer:"

        start_time = time.time()

        # Select model
        prediction_text = run_deepseek(
            prompt,
            max_new_tokens=256,
        )[1]
        elapsed = time.time() - start_time
        total_time += elapsed
        pred = extract_predicted_number(prediction_text)
        results.append((idx + 1, q.strip(), pred, gold, prediction_text))
        # Print cascade output for every incorrect prediction
        if pred is None or pred != gold:
            print(f"[{idx + 1}] Incorrect prediction.", flush=True)
            print(f"Question: {q}", flush=True)
            print(f"Gold: {gold}, Pred: {pred}", flush=True)
            print("Cascade output:", flush=True)
            print(prediction_text, flush=True)
            print("", flush=True)

        if pred is not None and pred == gold:
            correct += 1

        progress = (idx + 1) / n_examples
        accuracy_running = correct / (idx + 1)
        print(f"{progress:.2%} complete | {accuracy_running:.2%} accuracy", flush=True)

    accuracy = correct / n_examples
    print(f"Accuracy: {correct}/{n_examples} = {accuracy:.2%}")
    average_time = total_time / n_examples
    print(f"Average inference time per example: {average_time:.4f} seconds")

    return average_time


if __name__ == "__main__":
    arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(arg)