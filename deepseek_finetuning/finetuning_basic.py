import multiprocessing
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables
MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
MODEL_SIZE = "1.5B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LEN = 512

# Loading the dataset

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
   MODEL_NAME, padding=True, truncation=True, max_length=MAX_SEQ_LEN
)