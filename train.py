import yaml
import torch
import argparse
import os
import getpass
from google.colab import userdata
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# --- SETUP & CONFIG LOADER ---
def load_config(config_path="config.yaml"):
    """Load and validate YAML configuration file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = ['model', 'dataset', 'training', 'lora', 'quantization']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config section: {key}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")

# Parse command line args
parser = argparse.ArgumentParser(description="Fine-tune LLM for Stable Diffusion prompt expansion")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
args, unknown = parser.parse_known_args()

cfg = load_config(args.config)

# --- ---
def format_instruction(sample):
    """
    Extracts simple subject from complex SD prompts in this dataset.
    
    This dataset has a specific structure:
    "subject with descriptors. scene details. style keywords, artist names"
    
    Strategy (in order of priority):
    1. Split on PERIOD (.) - most reliable for this dataset
    2. If no period, split on artist reference (" by ")
    3. If neither, split on style keywords (concept art, trending on, etc.)
    4. Fallback: take first 60% of prompt
    
    Examples:
    Input:  "young, curly haired Natalie Portman as medieval innkeeper in dark inn"
    Output: [full prompt with scene details, lighting, style, artists]
    
    Input:  "comic portrait of female necromancer with big eyes"
    Output: [full prompt with fine details, anime style, artists, trending]
    """
    full_prompt = sample[cfg['dataset']['input_column']]
    
    # Strategy 1: Split on first period (most common delimiter in this dataset)
    if '. ' in full_prompt:
        simple_subject = full_prompt.split('. ')[0].strip()
    
    # Strategy 2: No period? Look for artist references
    elif ' by ' in full_prompt.lower():
        idx = full_prompt.lower().index(' by ')
        simple_subject = full_prompt[:idx].strip()
    
    # Strategy 3: Look for style keywords
    else:
        style_keywords = [
            'concept art', 'digital art', 'matte painting', 'fantasy art',
            'trending on', 'artstation', 'octane render', 'unreal engine',
            'photorealistic', 'hyperrealistic', '8k', '4k'
        ]
        
        cutoff_idx = None
        for keyword in style_keywords:
            if keyword in full_prompt.lower():
                idx = full_prompt.lower().index(keyword)
                if cutoff_idx is None or idx < cutoff_idx:
                    cutoff_idx = idx
        
        if cutoff_idx:
            simple_subject = full_prompt[:cutoff_idx].strip()
        else:
            # Strategy 4: Fallback - take first 60% of prompt
            char_limit = int(len(full_prompt) * 0.6)
            simple_subject = full_prompt[:char_limit].strip()
    
    # Clean up trailing punctuation and whitespace
    simple_subject = simple_subject.rstrip('.,;:!? ').strip()
    
    # Safety check: ensure we have something meaningful
    if len(simple_subject) < 10:
        # Too short, take more context
        if '. ' in full_prompt:
            parts = full_prompt.split('. ')
            simple_subject = '. '.join(parts[:2]).strip()
        else:
            simple_subject = ' '.join(full_prompt.split()[:15])
        simple_subject = simple_subject.rstrip('.,;:!? ')
    
    # Format exactly how Llama 3 expects it (Chat Template)
    return {
        "text": (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Expand this into a detailed Stable Diffusion prompt: {simple_subject}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{full_prompt}<|eot_id|>"
        )
    }

print(f"Loading dataset: {cfg['dataset']['name']}...")
dataset = load_dataset(cfg['dataset']['name'], split=cfg['dataset']['split'])
dataset = dataset.map(format_instruction)

# Create train/validation split
validation_split = cfg['dataset'].get('validation_split', 0.1)
if validation_split > 0:
    print(f"Splitting dataset: {int((1-validation_split)*100)}% train, {int(validation_split*100)}% validation")
    dataset = dataset.train_test_split(test_size=validation_split, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
else:
    train_dataset = dataset
    eval_dataset = None

# Verify it looks correct
print("\n" + "="*80)
print("Sample Training Data Pair:")
print("="*80)
print(train_dataset[0]['text'])
print("="*80 + "\n")

# HUGGINGFACE AUTHENTICATION 
try:
    hf_token = userdata.get('HF_TOKEN')
    print(" Found HF_TOKEN in Colab Secrets.")
except Exception as e:
    print(f" Secrets failed ({e}). Switching to manual input.")
    print("Please paste your Hugging Face token below (it will be hidden):")
    hf_token = getpass.getpass()
    login(token=hf_token)
    print(" Logged in manually!")
# WANDB CONFIGURATION
if cfg.get('wandb', {}).get('enabled', True):
    print("Configuring Weights & Biases...")
    os.environ['WANDB_PROJECT'] = cfg.get('wandb', {}).get('project', 'sd-prompt-expansion')
    os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

#  MODEL & TOKENIZER
print(f"\nLoading Model: {cfg['model']['name']}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg['quantization']['load_in_4bit'],
    bnb_4bit_quant_type=cfg['quantization']['quant_type'],
    bnb_4bit_compute_dtype=getattr(torch, cfg['quantization']['compute_dtype']),
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    cfg['model']['name'],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

print(f"Loading Tokenizer: {cfg['model']['name']}...")
tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], trust_remote_code=True)

# Handle padding token properly
if tokenizer.pad_token is None:
    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print("Adding special [PAD] token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

tokenizer.padding_side = "right"

print(f"Pad token: {tokenizer.pad_token}")
print(f"Vocabulary size: {len(tokenizer)}")

# --- 6. TRAINING SETUP ---
print("\nConfiguring LoRA...")
peft_config = LoraConfig(
    lora_alpha=cfg['lora']['alpha'],
    lora_dropout=cfg['lora']['dropout'],
    r=cfg['lora']['r'],
    bias=cfg['lora']['bias'],
    task_type="CAUSAL_LM",
    target_modules=cfg['lora']['target_modules']
)

print("Configuring Training Arguments...")
sft_config = SFTConfig(
    output_dir=cfg['training']['output_dir'],
    num_train_epochs=cfg['training']['num_epochs'],
    per_device_train_batch_size=cfg['training']['batch_size'],
    gradient_accumulation_steps=cfg['training']['grad_accumulation'],
    optim="paged_adamw_32bit",
    save_steps=cfg['training']['save_steps'],
    logging_steps=cfg['training']['logging_steps'],
    learning_rate=cfg['training']['learning_rate'],
    weight_decay=cfg['training']['weight_decay'],
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb" if cfg.get('wandb', {}).get('enabled', True) else "none",
    # Add evaluation if we have validation data
    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=cfg['training']['save_steps'] if eval_dataset else None,
    load_best_model_at_end=True if eval_dataset else False,
    metric_for_best_model="eval_loss" if eval_dataset else None,
)

tokenizer.model_max_length = cfg['training']['max_seq_length']
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=sft_config,
)
# ---  EXECUTION ---
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Total training samples: {len(train_dataset)}")
if eval_dataset:
    print(f"Total validation samples: {len(eval_dataset)}")
print(f"Epochs: {cfg['training']['num_epochs']}")
print(f"Batch size: {cfg['training']['batch_size']}")
print(f"Gradient accumulation: {cfg['training']['grad_accumulation']}")
print(f"Effective batch size: {cfg['training']['batch_size'] * cfg['training']['grad_accumulation']}")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

#  MODEL TESTING ---
def test_model(model, tokenizer, test_prompts):
    """Quick sanity check after training"""
    print("\n" + "="*80)
    print("TESTING MODEL WITH SAMPLE PROMPTS")
    print("="*80)
    
    model.eval()
    for prompt in test_prompts:
        input_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Expand this into a detailed Stable Diffusion prompt: {prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        print(f"\n{'─'*80}")
        print(f"INPUT: {prompt}")
        print(f"{'─'*80}")
        print(f"OUTPUT: {response}")
        print(f"{'─'*80}")
    
    print("="*80 + "\n")

# Test with sample prompts
test_prompts = [
    "a cat",
    "cyberpunk city",
    "portrait of a warrior",
    "fantasy landscape"
]
test_model(trainer.model, tokenizer, test_prompts)

#  SAVE MODEL
print("Saving Model...")
output_path = cfg['model']['new_model_name']
trainer.model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Model saved to: {output_path}")

# SUMMARY
print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"✓ Model trained and saved to: {output_path}")
print(f"✓ Ready for inference or merging")
print("\nNext steps:")
print("1. Test the model with your own prompts")
print("2. Merge LoRA weights with base model (optional)")
print("3. Push to HuggingFace Hub (optional)")
print("="*80)