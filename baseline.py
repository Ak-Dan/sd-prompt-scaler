"""
Baseline Test - Evaluates UNTRAINED model's behavior

"""

import torch
import pandas as pd
import getpass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from google.colab import userdata


try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print(" Logged in to Hugging Face successfully!\n")
except Exception as e:
    print(f" Secrets failed ({e}). Switching to manual input.")
    print("Please paste your Hugging Face token below (it will be hidden):")
    hf_token = getpass.getpass()
    login(token=hf_token)
    print(" Logged in manually!")
# --- CONFIG ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 

TEST_PROMPTS = [
    "a cat",
    "a dragon",
    "a futuristic city",
    "a warrior in armor",
    "portrait of an astronaut",
    "cyberpunk street at night",
    "fantasy landscape with mountains",
    "steampunk airship in clouds",
]

# --- LOAD BASE MODEL ---
print("="*80)
print("BASELINE TEST - Untrained Model Evaluation")
print("="*80)
print(f"\nLoading base model: {MODEL_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("✅ Model loaded successfully!\n")

# --- RUN INFERENCE ---
results = []

print("="*80)
print("Running baseline inference...")
print("="*80)

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: '{prompt}'")
    
    formatted_input = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Expand this into a detailed Stable Diffusion prompt: {prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    clean_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Metrics
    input_len = len(prompt)
    output_len = len(clean_response)
    expansion_ratio = output_len / input_len if input_len > 0 else 0
    
    sd_keywords = ["detailed", "realistic", "render", "octane", "unreal", "8k", "4k", 
                   "lighting", "artstation", "concept art", "digital art", "photorealistic"]
    keywords_found = sum(1 for kw in sd_keywords if kw.lower() in clean_response.lower())
    
    print(f"  Output: {clean_response[:80]}{'...' if len(clean_response) > 80 else ''}")
    print(f"  Expansion: {expansion_ratio:.1f}x | SD Keywords: {keywords_found}/{len(sd_keywords)}")
    
    results.append({
        "Input": prompt,
        "Baseline_Output": clean_response,
        "Input_Length": input_len,
        "Output_Length": output_len,
        "Expansion_Ratio": f"{expansion_ratio:.2f}x",
        "SD_Keywords_Count": keywords_found,
    })

# --- ANALYSIS ---
df = pd.DataFrame(results)
avg_expansion = df['Output_Length'].mean() / df['Input_Length'].mean()
avg_keywords = df['SD_Keywords_Count'].mean()

print("\n" + "="*80)
print(f"BASELINE ANALYSIS SUMMARY")
print(f"Avg Expansion: {avg_expansion:.2f}x")
print(f"Avg Keywords: {avg_keywords:.1f}")
print("="*80)

# Save
df.to_csv("baseline_results.csv", index=False)
print(f"\n📊 Results saved to: baseline_results.csv")