import os
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

os.system("huggingface-cli login --token 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")
dataset = load_dataset("presencesw/complexquestion_2WIKIMQA_10")['train']
model_name = "bn22/Mistral-7B-Instruct-v0.1-sharded"

# load_model
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # device_map={"": 0}
    device_map="cuda:0"
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def decompose_complex(example):
    complex_question = example['complex_question']
    system_prompt = example['system_prompt']
    user_prompt = example['user_prompt']
    input_text = "<s> [INST] " + system_prompt + " [/INST]\n" + user_prompt
    # print(input_text)
    model_input = tokenizer(input_text, return_tensors="pt").to("cuda")

    base_model.eval()
    with torch.no_grad():
        output = tokenizer.decode(base_model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True)
    example["llm_output"] = output.split("#\n")[-1]
    # print(example["llm_output"])
    return example

new_dataset = dataset.map(decompose_complex)
new_dataset.push_to_hub(f"presencesw/complexquestion_2WIKIMQA_10_Mistral", private=False)