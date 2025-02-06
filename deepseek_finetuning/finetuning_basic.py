import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Global configuration variables
MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
MODEL_SIZE = "1.5B"
# Target modules for LoRA adaptation - these are the attention layers we'll modify
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LEN = 512  # Maximum sequence length for tokenization
LORA_R = 32        # LoRA rank - determines the size of low-rank matrices
LORA_ALPHA = 64    # LoRA scaling factor - affects the magnitude of updates

# Configure 4-bit quantization settings
# This reduces memory usage while maintaining model quality
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",              # Use normal float 4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    bnb_4bit_use_double_quant=True          # Enable double quantization for additional memory savings
)

# Load the base model with quantization settings
# device_map="auto" automatically handles GPU/CPU placement
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
# Initialize tokenizer with padding and truncation settings
tokenizer = AutoTokenizer.from_pretrained(
   MODEL_NAME,
   padding=True,           # Enable padding for batch processing
   truncation=True,        # Enable truncation for long sequences
   max_length=MAX_SEQ_LEN  # Set maximum sequence length
)
logger.info(f"Memory usage: {model.get_memory_footprint() / 1e9:,.1f} GB")

# Configure LoRA settings
# LoRA reduces the number of trainable parameters by using low-rank decomposition
lora_config = LoraConfig(
    r=LORA_R,                       # Rank of the update matrices
    lora_alpha=LORA_ALPHA,          # Scaling factor for the updates
    lora_dropout=0.1,               # Dropout probability for regularization
    target_modules=TARGET_MODULES,  # Which modules to apply LoRA to
    init_lora_weights="gaussian",   # Initialize weights using gaussian distribution
    task_type="CAUSAL_LM",          # Specify the task type (causal language modeling)
    inference_mode=False,           # Enable training mode
)
# Apply LoRA to the model
# This wraps the original model with LoRA layers while keeping base model frozen
peft_model = get_peft_model(model, lora_config)

# Print information about trainable parameters
# This shows how many parameters are being trained vs frozen
peft_model.print_trainable_parameters()
