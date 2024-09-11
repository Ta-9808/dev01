from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, HfArgumentParser
from transformers import TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel
from peft import prepare_model_for_kbit_training, get_peft_model

import os, torch
from datasets import load_dataset
from trl import SFTTrainer
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import re

from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("HF_ACCESS_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = False,
)
print(bnb_config)

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        #load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
)
print(model)

