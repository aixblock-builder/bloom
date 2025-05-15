import os

import torch
from huggingface_hub import HfFolder
from transformers import pipeline

# --------------------------------------------------------------------------
# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)

from huggingface_hub import login

login(token=hf_token)
model_id = "bigscience/bloomz-7b1-mt"


def load():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        print("CUDA is available.")

        _model = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=dtype,
            device_map="auto",
            max_new_tokens=256,
        )
    else:
        print("No GPU available, using CPU.")
        _model = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            max_new_tokens=256,
        )


load()
