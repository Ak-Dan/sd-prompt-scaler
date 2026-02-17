"""
Comparison Script - Baseline vs Fine-tuned Model
Loads both models and compares their outputs side-by-side
"""

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from google.colab import userdata
from huggingface_hub import login
import getpass
import os
# ==============================================================================
# AUTHENTICATION
# ==============================================================================
try:
    hf_token = userdata.get('HF_TOKEN')
    login(token=hf_token)
    print(" Logged in to Hugging Face successfully!")
except Exception as e:
    print(f" Secrets failed ({e}). Switching to manual input.")
    print("Please paste your Hugging Face token below (it will be hidden):")
    hf_token = getpass.getpass()
    login(token=hf_token)
    print(" Logged in manually!")

# ==============================================================================
# CONFIGURATION 
# ==============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "/content/sd-prompt-expander-v1" 

# Verify the path exists before running
if not os.path.exists(ADAPTER_PATH):
    print(f" WARNING: Adapter folder '{ADAPTER_PATH}' not found in current directory.")
    print("Listing current directory contents:")
    print(os.listdir("."))
    print("Please update ADAPTER_PATH variable to match your saved folder name.")

# Test prompts
TEST_PROMPTS = [
    "a cat",
    "a dragon", 
    "a futuristic city",
    "a warrior in armor",
    "portrait of an astronaut",
    "cyberpunk street at night",
]

# --- HELPER FUNCTION ---
def generate_response(model, tokenizer, prompt):
    """Generate response for a given prompt"""
    formatted_input = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Expand this into a detailed Stable Diffusion prompt: {prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response

# --- LOAD MODELS ---
print("="*80)
print("BASELINE vs FINE-TUNED COMPARISON")
print("="*80)

print(f"\n[1/4] Loading base model: {BASE_MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"[2/4] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[3/4] Creating baseline model reference")
baseline_model = base_model
baseline_model.eval()

print(f"[4/4] Loading fine-tuned adapter from: {ADAPTER_PATH}")
try:
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    finetuned_model.eval()
    print(" Adapter loaded successfully!\n")
except Exception as e:
    print(f" Error loading adapter: {e}")
    print("\nNote: Run this script AFTER training is complete")
    print("If you haven't trained yet, run: python train.py --config config.yaml")
    exit(1)

# --- RUN COMPARISON ---
print("="*80)
print("RUNNING COMPARISON")
print("="*80)

results = []
sd_keywords = ["detailed", "realistic", "render", "octane", "unreal", "8k", "4k",
               "lighting", "artstation", "concept art", "digital art", "photorealistic",
               "trending", "fantasy", "professional", "epic", "dramatic"]

for i, prompt in enumerate(TEST_PROMPTS, 1):
    print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: '{prompt}'")
    print("─"*80)
    
    # Get baseline response
    print("Generating baseline response...")
    baseline_response = generate_response(baseline_model, tokenizer, prompt)
    
    # Get fine-tuned response  
    print("Generating fine-tuned response...")
    finetuned_response = generate_response(finetuned_model, tokenizer, prompt)
    
    # Calculate metrics
    baseline_len = len(baseline_response)
    finetuned_len = len(finetuned_response)
    input_len = len(prompt)
    
    baseline_ratio = baseline_len / input_len
    finetuned_ratio = finetuned_len / input_len
    
    baseline_keywords = sum(1 for kw in sd_keywords if kw.lower() in baseline_response.lower())
    finetuned_keywords = sum(1 for kw in sd_keywords if kw.lower() in finetuned_response.lower())
    
    # Improvement metrics
    expansion_improvement = finetuned_ratio - baseline_ratio
    keyword_improvement = finetuned_keywords - baseline_keywords
    
    print(f"\nBASELINE  ({baseline_ratio:.1f}x, {baseline_keywords} keywords):")
    print(f"  {baseline_response[:100]}{'...' if len(baseline_response) > 100 else ''}")
    
    print(f"\nFINE-TUNED ({finetuned_ratio:.1f}x, {finetuned_keywords} keywords):")
    print(f"  {finetuned_response[:100]}{'...' if len(finetuned_response) > 100 else ''}")
    
    print(f"\nIMPROVEMENT: {expansion_improvement:+.1f}x expansion, {keyword_improvement:+d} keywords")
    
    results.append({
        "Input": prompt,
        "Baseline_Output": baseline_response,
        "Finetuned_Output": finetuned_response,
        "Baseline_Expansion": f"{baseline_ratio:.2f}x",
        "Finetuned_Expansion": f"{finetuned_ratio:.2f}x",
        "Baseline_Keywords": baseline_keywords,
        "Finetuned_Keywords": finetuned_keywords,
        "Expansion_Improvement": f"{expansion_improvement:+.2f}x",
        "Keyword_Improvement": f"{keyword_improvement:+d}",
        "Better": "Fine-tuned" if finetuned_ratio > baseline_ratio and finetuned_keywords > baseline_keywords else "Baseline"
    })

# --- ANALYSIS ---
print("\n" + "="*80)
print("AGGREGATE ANALYSIS")
print("="*80)

df = pd.DataFrame(results)

# Calculate averages
avg_baseline_expansion = sum(float(r['Baseline_Expansion'].rstrip('x')) for r in results) / len(results)
avg_finetuned_expansion = sum(float(r['Finetuned_Expansion'].rstrip('x')) for r in results) / len(results)
avg_baseline_keywords = sum(r['Baseline_Keywords'] for r in results) / len(results)
avg_finetuned_keywords = sum(r['Finetuned_Keywords'] for r in results) / len(results)

print(f"\nExpansion Ratio:")
print(f"  Baseline:   {avg_baseline_expansion:.2f}x")
print(f"  Fine-tuned: {avg_finetuned_expansion:.2f}x")
print(f"  Improvement: {avg_finetuned_expansion - avg_baseline_expansion:+.2f}x ({(avg_finetuned_expansion/avg_baseline_expansion - 1)*100:.1f}% increase)")

print(f"\nSD Keywords Found:")
print(f"  Baseline:   {avg_baseline_keywords:.1f} keywords")
print(f"  Fine-tuned: {avg_finetuned_keywords:.1f} keywords")
print(f"  Improvement: {avg_finetuned_keywords - avg_baseline_keywords:+.1f} keywords ({(avg_finetuned_keywords/max(avg_baseline_keywords, 1) - 1)*100:.1f}% increase)")

# Success criteria
wins = sum(1 for r in results if r['Better'] == 'Fine-tuned')
print(f"\nFine-tuned wins: {wins}/{len(results)} ({wins/len(results)*100:.1f}%)")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if wins >= len(results) * 0.9 and avg_finetuned_expansion > avg_baseline_expansion * 1.5:
    print(" EXCELLENT: Fine-tuning produced significant improvements!")
    print("   → Model is ready for production use")
elif wins >= len(results) * 0.7:
    print(" GOOD: Fine-tuning improved performance")
    print("   → Model is functional, consider more training for better results")
elif wins >= len(results) * 0.5:
    print("  MODERATE: Fine-tuning showed some improvements")
    print("   → Consider adjusting hyperparameters or training longer")
else:
    print(" CONCERNING: Fine-tuning didn't improve performance")
    print("   → Check training logs, may need to retrain with different settings")

# --- SAVE RESULTS ---
df.to_csv("comparison_results.csv", index=False)
print(f"\n Detailed results saved to: comparison_results.csv")

# Show best examples
print("\n" + "="*80)
print("BEST EXAMPLES (for Stable Diffusion)")
print("="*80)

# Sort by expansion improvement
sorted_results = sorted(results, 
                       key=lambda x: float(x['Expansion_Improvement'].rstrip('x')), 
                       reverse=True)

for i, result in enumerate(sorted_results[:3], 1):
    print(f"\n{i}. INPUT: {result['Input']}")
    print(f"   BASELINE:   {result['Baseline_Output'][:80]}...")
    print(f"   FINE-TUNED: {result['Finetuned_Output']}")
    print(f"   IMPROVEMENT: {result['Expansion_Improvement']} expansion, {result['Keyword_Improvement']} keywords")

print("\n" + "="*80)
print("Comparison complete! ")
print("="*80)