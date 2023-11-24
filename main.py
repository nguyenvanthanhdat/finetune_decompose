import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
import torch
from peft import LoraConfig, PeftModel
from trl import SFTTrainer # For supervised finetuning

os.system("huggingface-cli login --token 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")
# dataset = load_dataset("presencesw/dataset_luat", use_auth_token=True)
dataset = load_dataset("presencesw/dataset_luat", token=True)
model_name = "vilm/vietcuna-7b-v3"
new_model = "vietcuna-7b-v3_luat"

# load_model
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25
max_seq_length = None
packing = False
# device_map = {"": 0}
device_map = "cuda:0"

# Load the base model with QLoRA configuration
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
    device_map="cuda:0",
    trust_remote_code=True,
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Load MistralAI tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# prompt_template = """Phân tách một câu hỏi phức tạp thành các câu hỏi đơn giản.
# Với mỗi câu hỏi đơn giản được tạo ra sẽ thể hiện cho mỗi vấn đề phụ từ câu hỏi phức tạp vì thế sau khi trả lời chúng ta sẽ dùng sự hiểu biết để trả lời câu hỏi phức tạp.
# Câu hỏi phức tạp:
# {complex_question}
# Câu hỏi đơn giản:"""

prompt_template = """Bạn là một luật sư Việt Nam. Bạn hãy phân tách câu hỏi phức tạp thành các câu hỏi đơn giản Với mỗi câu hỏi đơn giản được tạo ra sẽ thể hiện cho mỗi vấn đề phụ từ câu hỏi phức tạp vì thế sau khi trả lời chúng ta sẽ dùng sự hiểu biết pháp luật để trả lời câu hỏi phức tạp.

### Human:
Câu hỏi phức tạp:
{complex_question}
Câu hỏi đơn giản:
"""

def transform(examples):
    simple_question_lst = [[t["question"] for t in triple] for triple in examples["triplets"]]
    text = [prompt_template.format(complex_question=cq)+ "\n\n### Assistant:\n" + "\n".join(sq)
            for cq, sq in zip(examples["complex_question"], simple_question_lst)]
    examples["text"] = text
    return examples
dataset = dataset.map(transform, batched=True)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    do_train=True,
    do_eval=True,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    # max_steps=100, # the number of training steps the model will take
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    load_best_model_at_end=True,
    evaluation_strategy="steps"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Empty VRAM
import gc
del base_model
gc.collect()

del trainer
gc.collect()

torch.cuda.empty_cache() # PyTorch thing
gc.collect()
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}, 
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Push the model and tokenizer to the Hugging Face Model Hub
merged_model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)