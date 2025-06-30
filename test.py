from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from utils.generate import generate
import torch
import re
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLaDA
llada_checkpoint = "GSAI-ML/LLaDA-8B-Instruct"
llada_tokenizer = AutoTokenizer.from_pretrained(llada_checkpoint, trust_remote_code=True)
llada_model = (
    AutoModelForCausalLM.from_pretrained(llada_checkpoint, trust_remote_code=True).to(device).eval()
)

# # Load Dream
# dream_checkpoint = "Dream-org/Dream-v0-Instruct-7B"
# dream_tokenizer = AutoTokenizer.from_pretrained(dream_checkpoint, trust_remote_code=True)
# dream_model = (
#     AutoModel.from_pretrained(
#         dream_checkpoint,
#         trust_remote_code=True
#     )
#     .to(device)
#     .eval()
# )

# # Load DeepSeek
# ds_checkpoint = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# ds_tokenizer = AutoTokenizer.from_pretrained(
#     ds_checkpoint,
#     trust_remote_code=True,
# )
# ds_model = AutoModelForCausalLM.from_pretrained(
#     ds_checkpoint,
#     device_map="auto",
#     low_cpu_mem_usage=True
# )

# Prompt Builder
def build_thought_prompt(question: str) -> str:
    return (
        "Provide ONLY the step‑by‑step thought process for answering a question. "
        "List each step on its own line, numbered beginning with 1. "
        "Do not state the final answer or add any extra commentary.\n\n"
        # Example
        "Example\n"
        "Prompt: If I have 3 apples and my brother has 4 apples, how many do we have?\n"
        "Response:\n"
        "1. The user has 3 apples\n"
        "2. Their brother has 4 apples\n"
        "3. Together they must have 3 + 4 apples\n\n"
        # Actual question
        f"Prompt: {question}\n"
        "Response:"
    )

# Run Models
def run_llada(question: str, max_new_tokens: int = 256) -> str:

    m = [{"role": "user", "content": question},]
    user_input = llada_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = llada_tokenizer(user_input)['input_ids']
    input_ids = torch.tensor(inputs).to(device).unsqueeze(0)
    llada_model.config.use_cache = False
    out = generate(llada_model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    return llada_tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

def run_dream(question: str, max_new_tokens: int = 256, steps: int = 512) -> str:
    prompt = build_thought_prompt(question)
    messages = [{"role": "user", "content": prompt}]

    inputs = dream_tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = dream_model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=True,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    )

    generations = [
        dream_tokenizer.decode(g[len(p):].tolist())
        for p, g in zip(input_ids, output.sequences)
    ]
    gen = generations[0]
    if dream_tokenizer.eos_token:
        gen = gen.split(dream_tokenizer.eos_token)[0]
    return gen.strip()

def run_deepseek(prompt: str, max_new_tokens: int = 256) -> str:
    ds_model.eval()

    inputs = ds_tokenizer(prompt, return_tensors="pt").to(ds_model.device)
    output_ids = ds_model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = ds_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "</think>" in decoded:
        answer_part = decoded.split("</think>", 1)[1].strip()
    else:
        answer_part = decoded.strip()

    answer_sentence = answer_part.strip()

    result_text = decoded.strip()
    if not result_text.endswith("\n"):
        result_text += "\n"
    return f"{result_text}answer: {answer_sentence}"

# Extract thought process
def extract_thought_process(output: str) -> str:
    idx = output.find(':')
    if idx == -1:
        return output.strip()

    remainder = output[idx + 1 :].lstrip()
    collected = []
    for line in remainder.splitlines():
        if line.strip() == '':
            break
        collected.append(line)
    return '\n'.join(collected).strip()

def wrap_with_output(prompt: str, model_output: str) -> str:
    thought_process = extract_thought_process(model_output)
    return (
        f"{prompt}\n"
        f"<think>\n{thought_process}\n</think>\n"
    )

# Cascade Functions
def cascade_dream_to_deepseek(
    question: str,
    max_thought_tokens: int = 256,
    max_answer_tokens: int = 256,
    dream_steps: int = 512,
) -> str:

    raw_thoughts = run_dream(
        question,
        max_new_tokens=max_thought_tokens,
        steps=dream_steps,
    )
    wrapped_prompt = wrap_with_output(question, raw_thoughts)
    return run_deepseek(wrapped_prompt, max_new_tokens=max_answer_tokens)

def cascade_llada_to_deepseek(
    question: str,
    max_thought_tokens: int = 256,
    max_answer_tokens: int = 256,
) -> str:
    raw_thoughts = run_llada(question, max_new_tokens=max_thought_tokens)
    wrapped_prompt = wrap_with_output(question, raw_thoughts)
    return run_deepseek(wrapped_prompt, max_new_tokens=max_answer_tokens)

question = "If I have 4 friends, how many cookies do i need to give each friend 3 cookies?"
print(run_llada(question))