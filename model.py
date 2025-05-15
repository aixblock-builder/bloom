# model_marketplace.config
# {
#     "token_length": "128000",
#     "sampling_frequency": "44100",
#     "framework": "transformers",
#     "dataset_format": "llm",
#     "dataset_sample": "[id on s3]",
#     "weights": [
#         {
#             "name": "bloomz-7b1-mt",
#             "value": "bigscience/bloomz-7b1-mt",
#             "size": 25,
#             "parameters": "4b",
#             "tflops": 11,
#             "vram": 11,
#             "nodes": 1,
#         },
#     ],
#     "cuda": "12.6",
#     "task": [
#         "text-generation",
#         "question-answering",
#         "image-text-to-text",
#     ],
# }
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from typing import Iterator

import gradio as gr
import spaces
import torch
from aixblock_ml.model import AIxBlockMLBase
from huggingface_hub import HfFolder, login
from loguru import logger
from mcp.server.fastmcp import FastMCP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from function_ml import connect_project, download_dataset, upload_checkpoint
from logging_class import start_queue, write_log
from prompt import qa_without_context

# ------------------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN", "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")
HfFolder.save_token(hf_token)


hf_access_token = "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"
login(token=hf_access_token)
CUDA_VISIBLE_DEVICES = []
for i in range(torch.cuda.device_count()):
    CUDA_VISIBLE_DEVICES.append(i)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    f"{i}" for i in range(len(CUDA_VISIBLE_DEVICES))
)
print(os.environ["CUDA_VISIBLE_DEVICES"])


HOST_NAME = os.environ.get("HOST_NAME", "https://dev-us-west-1.aixblock.io")
TYPE_ENV = os.environ.get("TYPE_ENV", "DETECTION")


mcp = FastMCP("aixblock-mcp")

CHANNEL_STATUS = {}


class MyModel(AIxBlockMLBase):

    @mcp.tool()
    def action(self, command, **kwargs):
        """
        Execute commands for model operations, including shell command execution and model training.

        Args:
            command (str): The command to execute. Supported commands:
                - 'execute': Run a shell command
                - 'train': Start model training process
                - 'predict': Run model inference
                - 'tensorboard': Launch TensorBoard visualization
            **kwargs: Variable keyword arguments including:
                For 'execute' command:
                    - shell (str): The shell command to execute
                For 'train' command:
                    - model_id (str): Model identifier (default: 'bigscience/bloomz-7b1-mt')
                    - dataset_id (str): Dataset identifier
                    - push_to_hub (bool): Whether to push to HuggingFace Hub (default: True)
                    - hf_model_id (str): HuggingFace model ID
                    - push_to_hub_token (str): HuggingFace authentication token
                    - framework (str): Training framework (default: 'huggingface')
                    - task (str): Training task type (default: 'text-generation')
                    - trainingArguments (dict): Training configuration parameters
                    - cuda_debug (bool): Enable CUDA debugging (default: False)
                For 'predict' command:
                    - input_text (str): Text input for inference
                    - max_length (int): Maximum length of generated text (default: 512)
                    - temperature (float): Sampling temperature (default: 0.7)
                For 'tensorboard' command:
                    - logdir (str): Directory containing TensorBoard logs
                    - port (int): Port to run TensorBoard server (default: 6006)

        Returns:
            dict: A dictionary containing operation status or results
        """
        print(
            f"""
                command: {command}
            """
        )
        if command.lower() == "execute":
            _command = kwargs.get("shell", None)
            logger.info(f"Executing command: {_command}")
            subprocess.Popen(
                _command,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
            )
            return {"message": "command completed successfully"}

        elif command.lower() == "train":

            model_id = kwargs.get("model_id", "bigscience/bloomz-7b1-mt")
            dataset_id = kwargs.get(
                "dataset_id", "autoprogrammer/Qwen2.5-Coder-7B-Instruct-codeguardplus"
            )

            push_to_hub = kwargs.get("push_to_hub", True)
            hf_model_id = kwargs.get("hf_model_id", "bigscience/bloomz-7b1-mt")
            push_to_hub_token = kwargs.get(
                "push_to_hub_token", "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"
            )
            framework = kwargs.get("framework", "huggingface")
            task = kwargs.get("task", "text-generation")
            prompt = kwargs.get("prompt", "")
            trainingArguments = kwargs.get("TrainingArguments", None)
            cuda_debug = kwargs.get("cuda_debug", False)

            json_file = "training_args.json"
            absolute_path = os.path.abspath(json_file)

            with open(absolute_path, "w") as f:
                json.dump(trainingArguments, f)
            print(trainingArguments)

            if cuda_debug == True:
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
                os.environ["NCCL_DEBUG"] = "INFO"

            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["TORCH_USE_CUDA_DSA"] = "0"
            clone_dir = os.path.join(os.getcwd())
            project_id = kwargs.get("project_id", 0)
            token = kwargs.get("token", "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")
            checkpoint_version = kwargs.get("checkpoint_version")
            checkpoint_id = kwargs.get("checkpoint")
            dataset_version = kwargs.get("dataset_version")
            dataset_id = kwargs.get("dataset")
            channel_log = kwargs.get("channel_log", "training_logs")
            world_size = kwargs.get("world_size", 1)
            rank = kwargs.get("rank", 0)
            master_add = kwargs.get("master_add", "127.0.0.1")
            master_port = kwargs.get("master_port", "23456")
            host_name = kwargs.get("host_name", HOST_NAME)
            instruction_field = kwargs.get("prompt_field", "prompt")
            input_field = kwargs.get("input_field", "task_description")
            output_field = kwargs.get("output_field", "response")
            log_queue, logging_thread = start_queue(channel_log)
            write_log(log_queue)
            channel_name = f"{hf_model_id}_{str(uuid.uuid4())[:8]}"
            username = ""
            hf_model_name = ""

            # try:
            #     headers = {"Authorization": f"Bearer {push_to_hub_token}"}
            #     response = requests.get(
            #         "https://huggingface.co/api/whoami-v2", headers=headers
            #     )

            #     if response.status_code == 200:
            #         data = response.json()
            #         username = data.get("name")
            #         hf_model_name = f"{username}/{hf_model_id}"
            #         print(f"Username: {username}")
            #     else:
            #         print(f"Error: {response.status_code} - {response.text}")
            #         hf_model_name = "Token not correct"
            # except Exception as e:
            #     hf_model_name = "Token not correct"
            #     print(e)
            CHANNEL_STATUS[channel_name] = {
                "status": "training",
                "hf_model_id": hf_model_name,
                "command": command,
                "created_at": time.time(),
            }
            print(f"🚀 Đã bắt đầu training kênh: {channel_name}")

            def func_train_model(
                clone_dir,
                project_id,
                token,
                checkpoint_version,
                checkpoint_id,
                dataset_version,
                dataset_id,
                model_id,
                world_size,
                rank,
                master_add,
                master_port,
                prompt,
                json_file,
                channel_log,
                hf_model_id,
                push_to_hub,
                push_to_hub_token,
                host_name,
            ):

                dataset_path = None
                project = connect_project(host_name, token, project_id)

                if dataset_version and dataset_id and project:
                    dataset_path = os.path.join(
                        clone_dir, f"datasets/{dataset_version}"
                    )

                    if not os.path.exists(dataset_path):
                        data_path = os.path.join(clone_dir, "data_zip")
                        os.makedirs(data_path, exist_ok=True)

                        dataset_name = download_dataset(project, dataset_id, data_path)
                        print(dataset_name)
                        if dataset_name:
                            data_zip_dir = os.path.join(data_path, dataset_name)

                            with zipfile.ZipFile(data_zip_dir, "r") as zip_ref:
                                zip_ref.extractall(dataset_path)

                            extracted_files = os.listdir(dataset_path)
                            zip_files = [
                                f for f in extracted_files if f.endswith(".zip")
                            ]

                            if len(zip_files) == 1:
                                inner_zip_path = os.path.join(
                                    dataset_path, zip_files[0]
                                )
                                print(
                                    f"🔁 Found inner zip file: {inner_zip_path}, extracting..."
                                )
                                with zipfile.ZipFile(inner_zip_path, "r") as inner_zip:
                                    inner_zip.extractall(dataset_path)
                                os.remove(inner_zip_path)

                subprocess.run(
                    ("whereis accelerate"),
                    shell=True,
                )
                print("===Train===")
                if framework == "huggingface":
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            print("master node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                head_node_ip=master_add,
                                port=master_port,
                                machine_rank=rank,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.run(
                                command,
                                shell=True,
                            )

                    else:
                        if torch.cuda.device_count() > 1:  # multi gpu
                            command = (
                                "venv/bin/accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                num_processes=world_size * torch.cuda.device_count(),
                                SLURM_NNODES=world_size,
                                # head_node_ip=os.environ.get("head_node_ip", master_add),
                                port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print("================2")
                            print(command)
                            print("================2")
                            process = subprocess.run(command, shell=True)

                        elif torch.cuda.device_count() == 1:  # one gpu
                            command = (
                                "venv/bin/accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print("================")
                            print(command)
                            print("================")
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                        else:  # no gpu
                            command = (
                                "venv/bin/accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                            while True:
                                output = process.stdout.readline().decode("utf-8")
                                if output == "" and process.poll() is not None:
                                    break
                                if output:
                                    print(output, end="")
                            process.wait()

                elif framework == "pytorch":
                    process = subprocess.run(
                        ("whereis torchrun"),
                        shell=True,
                    )

                    if int(world_size) > 1:
                        if rank == 0:
                            print("master node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr="127.0.0.1",
                                master_port="23456",
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                        else:
                            print("worker node")
                            command = (
                                "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                            ).format(
                                nnodes=int(world_size),
                                node_rank=int(rank),
                                nproc_per_node=world_size * torch.cuda.device_count(),
                                master_addr=master_add,
                                master_port=master_port,
                                file_name="./run_distributed_accelerate.py",
                                json_file=json_file,
                                dataset_path=dataset_path,
                                channel_log=channel_log,
                                hf_model_id=hf_model_id,
                                push_to_hub=push_to_hub,
                                model_id=model_id,
                                push_to_hub_token=push_to_hub_token,
                                instruction_field=instruction_field,
                                input_field=input_field,
                                output_field=output_field,
                            )
                            print(command)
                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                text=True,
                            )
                    else:
                        command = (
                            "venv/bin/torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                            "{file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --model_id {model_id} --instruction_field {instruction_field} --input_field {input_field} --output_field {output_field}"
                        ).format(
                            nnodes=int(world_size),
                            node_rank=int(rank),
                            nproc_per_node=world_size * torch.cuda.device_count(),
                            file_name="./run_distributed_accelerate.py",
                            json_file=json_file,
                            dataset_path=dataset_path,
                            channel_log=channel_log,
                            hf_model_id=hf_model_id,
                            push_to_hub=push_to_hub,
                            model_id=model_id,
                            push_to_hub_token=push_to_hub_token,
                            instruction_field=instruction_field,
                            input_field=input_field,
                            output_field=output_field,
                        )
                        process = subprocess.run(
                            command,
                            shell=True,
                        )
                CHANNEL_STATUS[channel_name]["status"] = "done"
                output_dir = "./data/checkpoint"
                print(push_to_hub)
                if push_to_hub:
                    import datetime

                    output_dir = "./data/checkpoint"
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")
                    version = f"{date_str}-{time_str}"

                    upload_checkpoint(project, version, output_dir)

            train_thread = threading.Thread(
                target=func_train_model,
                args=(
                    clone_dir,
                    project_id,
                    token,
                    checkpoint_version,
                    checkpoint_id,
                    dataset_version,
                    dataset_id,
                    model_id,
                    world_size,
                    rank,
                    master_add,
                    master_port,
                    prompt,
                    absolute_path,
                    channel_log,
                    hf_model_id,
                    push_to_hub,
                    push_to_hub_token,
                    host_name,
                ),
            )
            train_thread.start()

            return {
                "message": "train completed successfully",
                "channel_name": channel_name,
            }
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "./inference/generate.py"])
            return {"message": "train stop successfully", "result": "Done"}

        elif command.lower() == "tensorboard":

            def run_tensorboard():
                p = subprocess.Popen(
                    f"tensorboard --logdir /app/data/checkpoint/runs --host 0.0.0.0 --port=6006",
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                )
                out = p.communicate()
                print(out)

            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}

        elif command.lower() == "predict":
            prompt = kwargs.get("prompt", None)
            model_id = kwargs.get("model_id", "bigscience/bloomz-7b1-mt")
            text = kwargs.get("text", None)
            token_length = kwargs.get("token_lenght", 30)
            task = kwargs.get("task", "")
            voice = kwargs.get("voice", "")
            max_new_token = kwargs.get("max_new_token", 256)
            temperature = kwargs.get("temperature", 0.7)
            top_k = kwargs.get("top_k", 50)
            top_p = kwargs.get("top_p", 0.95)

            predictions = []

            if not prompt or prompt == "":
                prompt = text

            from huggingface_hub import login

            hf_access_token = kwargs.get(
                "hf_access_token", "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"
            )
            login(token=hf_access_token)

            def smart_pipeline(
                model_id: str,
                token: str,
                local_dir="./data/checkpoint",
                task="text-generation",
            ):
                try:
                    import os

                    model_name = model_id.split("/")[-1]
                    local_model_dir = os.path.join(local_dir, model_name)
                    if os.path.exists(local_model_dir) and os.path.exists(
                        os.path.join(local_model_dir, "config.json")
                    ):
                        print(f"✅ Loading model from local: {local_model_dir}")
                        model_source = local_model_dir
                    else:
                        print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                        model_source = model_id
                except:
                    print(f"☁️ Loading model from HuggingFace Hub: {model_id}")
                    model_source = model_id

                # Xác định dtype và device
                if torch.cuda.is_available():
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16

                    print("Using CUDA.")
                    pipe = pipeline(
                        task,
                        model=model_source,
                        torch_dtype=dtype,
                        device_map="auto",
                        token=token,
                        max_new_tokens=256,
                    )
                else:
                    print("Using CPU.")
                    pipe = pipeline(
                        task,
                        model=model_source,
                        device_map="cpu",
                        token=token,
                        max_new_tokens=256,
                    )

                return pipe

            _model = smart_pipeline(model_id, hf_access_token)
            generated_text = qa_without_context(_model, prompt)

            print(generated_text)
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": "generated_text",
                            "to_name": "text_output",
                            "type": "textarea",
                            "value": {"text": [generated_text]},
                        }
                    ],
                    "model_version": "",
                }
            )

            return {"message": "predict completed successfully", "result": predictions}
        elif command.lower() == "prompt_sample":
            task = kwargs.get("task", "")
            if task == "question-answering":
                prompt_text = f"""
                    Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
            elif task == "text-classification":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

            elif task == "summarization":
                prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
            return {
                "message": "prompt_sample completed successfully",
                "result": prompt_text,
            }

        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}

        elif command == "status":
            channel = kwargs.get("channel", None)

            if channel:
                # Nếu có truyền kênh cụ thể
                status_info = CHANNEL_STATUS.get(channel)
                if status_info is None:
                    return {"channel": channel, "status": "not_found"}
                elif isinstance(status_info, dict):
                    return {"channel": channel, **status_info}
                else:
                    return {"channel": channel, "status": status_info}
            else:
                # Lấy tất cả kênh
                if not CHANNEL_STATUS:
                    return {"message": "No channels available"}

                channels = []
                for ch, info in CHANNEL_STATUS.items():
                    if isinstance(info, dict):
                        channels.append({"channel": ch, **info})
                    else:
                        channels.append({"channel": ch, "status": info})

                return {"channels": channels}
        else:
            return {"message": "command not supported", "result": None}

    @mcp.tool()
    def model(self, **kwargs):
        """
        This tool demos a code model with 7B parameters fine-tuned for chat instructions.
        You can interact with the model by sending a message and the model will generate a response based on the input.
        The model is loaded from huggingface hub and can be customized by passing the model name and other parameters.
        For example, you can try the 7B model in the official homepage.

        Args:
            model_id (str, optional): The model id to load from huggingface hub. Defaults to "bigscience/bloomz-7b1-mt".
            project_id (int, optional): The project id to use for the gradio app. Defaults to 0.
            hf_access_token (str, optional): The huggingface access token to use for loading the model. Defaults to "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".

        Returns:
            dict: A dictionary containing the share url and local url of the gradio app.
        """
        global model_demo, tokenizer_demo, model_loaded_demo, model_id_demo

        model_id_demo = kwargs.get("model_id", "bigscience/bloomz-7b1-mt")
        project_id = kwargs.get("project_id", 0)

        print(
            f"""\
        Project ID: {project_id}
        """
        )
        from huggingface_hub import login

        hf_access_token = kwargs.get(
            "hf_access_token", "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"
        )
        login(token=hf_access_token)
        MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

        DESCRIPTION = """\
        # Bloom
        This space demonstrates model [Bloom](https://huggingface.co/bigscience/bloomz-7b1-mt), a code model fine-tuned for chat instructions.
        """

        if not torch.cuda.is_available():
            DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model_demo = None
        tokenizer_demo = None
        model_loaded_demo = False

        def load_model(model_id):
            global model_demo, tokenizer_demo, model_loaded_demo
            if torch.cuda.is_available() and not model_loaded_demo:
                model_demo = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    token=hf_access_token,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    torch_dtype=compute_dtype,
                )
                tokenizer_demo = AutoTokenizer.from_pretrained(
                    model_id, token=hf_access_token
                )
                tokenizer_demo.use_default_system_prompt = False
                model_loaded_demo = True
                return f"Model {model_id} loaded successfully!"
            elif model_loaded_demo:
                return "Model is already loaded! Please refresh the page to load a different model."
            else:
                return "Error: CUDA is not available!"

        @spaces.GPU
        def generate(
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.6,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1,
        ) -> Iterator[str]:
            if not model_loaded_demo:
                return (
                    "Please load the model first by clicking the 'Load Model' button."
                )
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": str(system_prompt)})

            # Add chat history
            for user_msg, assistant_msg in chat_history:
                chat_messages.append({"role": "user", "content": str(user_msg)})
                chat_messages.append(
                    {"role": "assistant", "content": str(assistant_msg)}
                )

            # Add the current message
            chat_messages.append({"role": "user", "content": str(message)})
            prompt = ""
            for message in chat_messages:
                if message["role"] == "user":
                    prompt += f"User: {message['content']}\n"
                else:
                    prompt += f"Assistant: {message['content']}\n"

            model_inputs = tokenizer_demo([prompt], return_tensors="pt").to(
                model_demo.device
            )
            if model_inputs.input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
                model_inputs.input_ids = model_inputs.input_ids[
                    :, -MAX_INPUT_TOKEN_LENGTH:
                ]
                gr.Warning(
                    f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens."
                )

            outputs = model_demo.generate(**model_inputs, max_new_tokens=512)
            response = tokenizer_demo.decode(outputs[0], skip_special_tokens=True)
            lines = response.strip().split("\n")
            last_message = None
            for line in reversed(lines):
                if line.startswith("Assistant:"):
                    last_message = line.replace("Assistant:", "").strip()
                    break
            return last_message

        chat_interface = gr.ChatInterface(
            fn=generate,
            stop_btn=gr.Button("Stop"),
            examples=[
                ["implement snake game using pygame"],
                [
                    "Can you explain briefly to me what is the Python programming language?"
                ],
                ["write a program to find the factorial of a number"],
            ],
        )

        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)
            with gr.Row():
                load_btn = gr.Button("Load Model")
                status_text = gr.Textbox(label="Model Status", interactive=False)
            load_btn.click(fn=lambda: load_model(model_id_demo), outputs=status_text)
            chat_interface.render()

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )
        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def model_trial(self, project, **kwargs):
        import gradio as gr

        return {"message": "Done", "result": "Done"}

        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np

            def predict(input_img):
                import cv2

                result = self.action(
                    project, "predict", collection="", data={"img": input_img}
                )
                print(result)
                if result["result"]:
                    boxes = result["result"]["boxes"]
                    names = result["result"]["names"]
                    labels = result["result"]["labels"]

                    for box, label in zip(boxes, labels):
                        box = [int(i) for i in box]
                        label = int(label)
                        input_img = cv2.rectangle(
                            input_img, box, color=(255, 0, 0), thickness=2
                        )
                        # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        input_img = cv2.putText(
                            input_img,
                            names[label],
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                return input_img

            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'

            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(
                    project, "train", collection="", data=dataset_choosen
                )
                return result["message"]

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os

                checkpoint_list = [
                    i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")
                ]
                checkpoint_list = [
                    f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>"
                    for i in checkpoint_list
                ]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [
                                i
                                for i in os.listdir(
                                    f"my_ml_backend/{project}/{folder}/weights"
                                )
                                if i.endswith(".pt")
                            ]
                            project_checkpoint_list = [
                                f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                                for i in project_checkpoint_list
                            ]
                            checkpoint_list.extend(project_checkpoint_list)

                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)

            def upload_file(file):
                return "File uploaded!"

            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])

                    gr.Interface(
                        predict,
                        gr.Image(
                            elem_classes=["upload_image"],
                            sources="upload",
                            container=False,
                            height=345,
                            show_label=False,
                        ),
                        gr.Image(
                            elem_classes=["upload_image"],
                            container=False,
                            height=345,
                            show_label=False,
                        ),
                        allow_flagging=False,
                    )

                # with gr.TabItem("Webcam", id=1):
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown(
                                "## Dataset template to prepare your own and initiate training"
                            )
                            with gr.Row():
                                # get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")
                                datasets = [
                                    (f"dataset{i}", name)
                                    for i, name in enumerate(os.listdir("./datasets"))
                                ]

                                dataset_choosen = gr.Dropdown(
                                    datasets,
                                    label="Choose dataset",
                                    show_label=False,
                                    interactive=True,
                                    type="value",
                                )
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML(
                                    """
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>"""
                                )

                                dataset_choosen.select(
                                    download_btn, None, download_link
                                )

                                # when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown(
                                "## Upload your sample dataset to have a trial training"
                            )
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(
                                predict,
                                gr.File(
                                    elem_classes=["upload_image"],
                                    file_types=["tar", "zip"],
                                ),
                                gr.Label(
                                    elem_classes=["upload_image"], container=False
                                ),
                                allow_flagging=False,
                            )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(
                                    trial_training, dataset_choosen, None
                                )

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(
            share=True,
            quiet=True,
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            show_error=True,
        )

        return {"share_url": share_url, "local_url": local_url}

    @mcp.tool()
    def download(self, project, **kwargs):
        from flask import request, send_from_directory

        file_path = request.args.get("path")
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
