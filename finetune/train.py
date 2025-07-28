import json
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForSeq2Seq
from modelscope import snapshot_download


#数据预处理
dataset = load_dataset('json', data_files={
    'train': 'train_alpaca.jsonl',
    'val': 'val_alpaca.jsonl'
})

 
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-1.5B",
    use_fast=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  

 
def process_func(example):
    # 构建指令格式
    prompt = f"<|im_start|>system\n你是一个智能助手<|im_end|>\n" \
             f"<|im_start|>user\n{example['instruction']}{example['input']}<|im_end|>\n" \
             f"<|im_start|>assistant\n"
    
    # 组合完整输入
    inputs = tokenizer(prompt, add_special_tokens=False)
    targets = tokenizer(example['output'], add_special_tokens=False)
    
    # 拼接输入输出
    input_ids = inputs["input_ids"] + targets["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"] + targets["attention_mask"] + [1]
    
    # 设置labels（仅计算输出部分的loss）
    labels = [-100] * len(inputs["input_ids"]) + targets["input_ids"] + [tokenizer.pad_token_id]
    
    MAX_LENGTH = 2048
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

 
tokenized_ds = dataset.map(
    process_func,
    remove_columns=dataset["train"].column_names
)
 
 
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
 
training_args = TrainingArguments(
    output_dir="./qwen2-finetune",

    per_device_train_batch_size=1,   
    gradient_accumulation_steps=4,  #4次step梯度累积  
    learning_rate=1e-4,             
    bf16=True,

    eval_strategy="steps",          #10 step计算一次loss
    eval_steps=10, 
    per_device_eval_batch_size=4,

    num_train_epochs=2,
    logging_steps=100,
    save_strategy="epoch",           #每个epoch保存一个checkpoint
    gradient_checkpointing=False,    #节省激活值所占显存
)

 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["val"],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()


model.save_pretrained("./qwen2-finetune-final")
tokenizer.save_pretrained("./qwen2-finetune-final")
