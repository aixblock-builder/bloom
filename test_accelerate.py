import torch
# from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from random import randint
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments #BitsAndBytesConfig,AutoConfig,
from trl import SFTTrainer 
# import pathlib
from transformers import TrainerCallback
import numpy as np
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding
import wandb
import argparse
import json
import inspect
from logging_class import start_queue, write_log, stop_log

# Tạo parser cho dòng lệnh
parser = argparse.ArgumentParser(description='AIxBlock')
parser.add_argument('--training_args_json', type=str, default=None, help="JSON string for training arguments")
parser.add_argument('--dataset_local', type=str, default=None, help="dataset id")
parser.add_argument('--channel_log', type=str, default=None, help="channel_log")
parser.add_argument('--hf_model_id', type=str, default=None, help="hf_model_id")
parser.add_argument('--push_to_hub', type=str, default=None, help="push_to_hub")
parser.add_argument('--push_to_hub_token', type=str, default=None, help="push_to_hub_token")
# Phân tích các tham số dòng lệnh
args = parser.parse_args()

log_queue, logging_thread = start_queue(args.channel_log)
write_log(log_queue)

dataset_local = args.dataset_local
is_use_local = False
num_train_epochs = 1
per_train_dataset = 0.8
per_test_dataset = 0.2
output_dir="./data/checkpoint"

push_to_hub= True if args.push_to_hub and args.push_to_hub == "True" else False
hf_model_id = args.hf_model_id if args.hf_model_id else "llama3"
push_to_hub_token = args.push_to_hub_token if args.push_to_hub_token else "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"

# Nếu có file JSON, đọc và phân tích nó
print(args.training_args_json)
if args.training_args_json:
    with open(args.training_args_json, 'r') as f:
        training_args_dict = json.load(f)
else:
    training_args_dict = {}

if not training_args_dict:
    training_args_dict = {}

wandb.login('allow',"69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
# Hugging Face model id

if "model_id" in training_args_dict and training_args_dict["model_id"] != "None":
    model_id = training_args_dict["model_id"] #"facebook/opt-125m" # or  `appvoid/llama-3-1b` tiiuae/falcon-7b` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
else:
    model_id = "facebook/opt-125m"

if not training_args_dict:
    training_args_dict = {}

dataset_id = "Sujithanumala/Llama_3.2_1B_IT_dataset"

if "dataset_id" in training_args_dict:
    dataset_id= training_args_dict["dataset_id"] #"Sujithanumala/Llama_3.2_1B_IT_dataset"
else:
    dataset_id = "Sujithanumala/Llama_3.2_1B_IT_dataset"

    if dataset_local and dataset_local!="None":
        dataset_id = dataset_local
        is_use_local = True

if "num_train_epochs" in training_args_dict:
    num_train_epochs = int(training_args_dict["num_train_epochs"])

if "per_train_dataset" in training_args_dict:
    per_train_dataset = int(training_args_dict["per_train_dataset"])

if "per_test_dataset" in training_args_dict:
    per_test_dataset = int(training_args_dict["per_test_dataset"])

if not is_use_local:
    dataset = load_dataset(dataset_id, split="train")
    # eval_dataset = train_dataset 
    train_test_split = dataset.train_test_split(test_size=per_test_dataset, seed=42)  # 20% cho eval
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
else:
    # Load dataset từ thư mục local
    dataset = load_dataset(
        "json", 
        data_files={
            "train": f"{dataset_id}/train_data.json",
            "test": f"{dataset_id}/test_data.json"
        }
    )

    # Truy cập từng tập dữ liệu
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

# train_valid = train_dataset['train'].train_test_split(test_size=0.2)
# test_valid = train_dataset['test'].train_test_split(test_size=0.2)
# train_dataset = train_valid['train']
# test_dataset = test_valid['test']
# print(f"train_dataset: {train_dataset}")
# print(f"test_dataset: {test_dataset}")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {}

                ### Input:
                {}

                ### Response:
                {}"""

# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs       = examples["input"]
#     outputs      = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return {"text": texts}
# def tokenizer_func(examples):
#     instructions = examples["instruction"]
#     inputs       = examples["input"]
#     outputs      = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return tokenizer("".join(texts),truncation=True, padding=True, max_length=128, return_tensors="pt")
#                 # examples follow format of resp json files
# train_dataset = train_dataset.map(tokenizer_func,remove_columns=train_dataset.column_names, batched=True)
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):  
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return tokenizer(texts,truncation=True, padding=True, max_length=128, return_tensors="pt")                         
    # inputs = [ex['en'] for ex in examples['translation']]      
    # targets = [ex['ru'] for ex in examples['translation']]
    # model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, padding=True, truncation=True)
tokenized_datasets = train_dataset.map(
    preprocess_function,
    batched=True,
    # remove_columns=train_dataset['train'].column_names,
)
# eval_tokenized_datasets = eval_dataset.map(
#     preprocess_function,
#     batched=True,
#     # remove_columns=train_dataset['train'].column_names,
# )

data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16,collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)
# for epoch in range(2):
#     model.train()
#     for step, batch in enumerate(train_dataloader):          
#         outputs = model(**batch)
#         loss = outputs.loss
# print(len(train_dataset))
# print(len(eval_dataset))

print("Data is formatted and ready!")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

class TrainOnStartCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        # Log training loss at step 0
        logs = logs or {}
        self.log(logs)

    def log(self, logs):
        print(f"Logging at start: {logs}")

def is_valid_type(value, expected_type):
    from typing import get_origin, get_args, Union
    # Nếu không có type hint (Empty), chấp nhận giá trị
    if expected_type is inspect._empty:
        return True
    # Nếu type hint là generic (Union, Optional, List, etc.)
    origin = get_origin(expected_type)
    if origin is Union:  # Xử lý Union hoặc Optional
        return any(is_valid_type(value, arg) for arg in get_args(expected_type))
    if origin is list:  # Xử lý List
        return isinstance(value, list) and all(is_valid_type(v, get_args(expected_type)[0]) for v in value)
    if origin is dict:  # Xử lý Dict
        key_type, value_type = get_args(expected_type)
        return (
            isinstance(value, dict) and
            all(is_valid_type(k, key_type) for k in value.keys()) and
            all(is_valid_type(v, value_type) for v in value.values())
        )
    # Kiểm tra kiểu cơ bản (int, float, str, etc.)
    return isinstance(value, expected_type)

print("===========")
# if training_args_dict:
#     # Kết hợp dictionary từ JSON (nếu có) và giá trị mặc định
#     training_args_values = {**training_args_dict}

#     param_annotations = inspect.signature(TrainingArguments.__init__).parameters

#     valid_args = set(param_annotations.keys())
#     filtered_args = {k: v for k, v in training_args_values.items() if k in valid_args}

#     # Kiểm tra định dạng giá trị
#     validated_args = {}
#     for k, v in filtered_args.items():
#         expected_type = param_annotations[k].annotation  # Lấy kiểu dữ liệu mong đợi
#         if is_valid_type(v, expected_type):
#             validated_args[k] = v
#         else:
#             print(f"Skipping invalid parameter: {k} (expected {expected_type}, got {type(v)})")

#     # Khởi tạo TrainingArguments với tham số đã được xác thực
#     training_args = TrainingArguments(**validated_args)

# else:
if not push_to_hub:
    training_args = TrainingArguments(
        output_dir= f"./data/checkpoint", # directory to save and repository id
        # logging_dir= '/app/data/logs',
        learning_rate=2e-4,
        log_level="debug",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch", 
        # report_to="tensorboard",
        # report_to="wandb",
        use_cpu=False,
        bf16 = False,
        fp16 = False
    )
else:
    training_args = TrainingArguments(
        output_dir= f"./data/checkpoint", # directory to save and repository id
        # logging_dir= '/app/data/logs',
        learning_rate=2e-4,
        log_level="debug",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch", 
        # report_to="tensorboard",
        # report_to="wandb",
        use_cpu=False,
        bf16 = False,
        fp16 = False,
        # report_to="tensorboard", #azure_ml, comet_ml, mlflow, neptune, tensorboard, wandb, codecarbon, clearml, dagshub, flyte, dvclive
        push_to_hub = push_to_hub,
        push_to_hub_model_id=hf_model_id,#[project-id]-[model-name]-[datetime]
        push_to_hub_token=push_to_hub_token,
    )
print(training_args)

# https://github.com/huggingface/accelerate/issues/2618
# https://github.com/huggingface/huggingface-llama-recipes/blob/main/fine_tune/qlora_405B.slurm
# https://gist.github.com/rom1504/474f97a95a526d40ae44a3fc3c657a2e
# https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh
# https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multigpu.sh
# https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode_fsdp.sh
# https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multicpu.sh
trainer = SFTTrainer(
    dataset_text_field = "text",
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    # eval_dataset=eval_tokenized_datasets,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
    dataset_kwargs={
                        "add_special_tokens": False,  # We template with special tokens
                        "append_concat_token": False, # No need to add additional separator token
                        'skip_prepare_dataset': True # skip the dataset preparation
                    },
    callbacks=[TrainOnStartCallback()]
)
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
if push_to_hub:
    trainer.push_to_hub()
# save model
# MODEL_DIR = os.getenv('MODEL_DIR', './data/checkpoint')
# FINETUNED_MODEL_NAME = os.getenv('FINETUNED_MODEL_NAME',hf_model_id)
# chk_path = str(pathlib.Path(MODEL_DIR) / FINETUNED_MODEL_NAME)
# print(f"Model is trained and saved as {chk_path}")
# trainer.save_model(chk_path)
# push to hub

# free the memory again
del model
del trainer
torch.cuda.empty_cache()