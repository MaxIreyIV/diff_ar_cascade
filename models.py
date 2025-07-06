from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from utils.generate import generate
import torch
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load Dream
dream_checkpoint = "Dream-org/Dream-v0-Instruct-7B"
dream_tokenizer = AutoTokenizer.from_pretrained(dream_checkpoint, trust_remote_code=True)
dream_model = (
    AutoModel.from_pretrained(
        dream_checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": 1}
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
    torch_dtype=torch.float16).to(device)

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
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = ds_tokenizer.apply_chat_template(messages,  
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True)
    model_inputs = ds_tokenizer([inputs], return_tensors="pt").to(ds_model.device)

    generated_ids = ds_model.generate(
             **model_inputs,
        do_sample=True,          # ← stays
        temperature=0.7,
        top_p=0.9,

        max_new_tokens=max_new_tokens,  # keep tight
        use_cache=True
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
    # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = ds_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = ds_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content

question = "What is 2 + 2?"
rest = run_deepseek(question)
print("thinking_content: ", rest[0])
print("content: ", rest[1])