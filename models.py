from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from utils.generate import generate
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Dream
dream_checkpoint = "Dream-org/Dream-v0-Instruct-7B"
dream_tokenizer = AutoTokenizer.from_pretrained(dream_checkpoint, trust_remote_code=True)
dream_model = (
    AutoModel.from_pretrained(
        dream_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 0}
    )
    .eval()
)

# Load DeepSeek
ds_checkpoint = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
ds_tokenizer = AutoTokenizer.from_pretrained(
    ds_checkpoint,
    trust_remote_code=True
)
ds_model = AutoModelForCausalLM.from_pretrained(
    ds_checkpoint,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Run Dream
def run_dream(question: str, max_new_tokens: int = 256, steps: int = 256) -> str:
    messages = [{"role": "user", "content": question}]
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
        dtype=torch.float16,
    )
    generations = [
    dream_tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

    return generations[0].split(dream_tokenizer.eos_token)[0]

def run_deepseek(prompt: str, max_new_tokens: int = 512) -> str:
    ds_model.eval()
    inputs = ds_tokenizer(prompt, return_tensors="pt").to(ds_model.device)
    output_ids = ds_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        use_cache=True
    )
    decoded = ds_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    result_text = decoded.strip()
    if not result_text.endswith("\n"):
        result_text += "\n"
    return result_text

question = "What is 2 + 2?"
print(run_deepseek(question))