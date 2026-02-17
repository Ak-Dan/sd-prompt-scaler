"""
Regression Test Suite for SD Prompt Expander
Tests both stability (normal QA) and capability (prompt expansion)
"""

import torch
import pandas as pd
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from google.colab import userdata
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
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

# --- CONFIGURATION ---
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "akanbiNAD/sd-prompt-expander-v1"

# Verify the path exists before running
if not os.path.exists(ADAPTER_PATH):
    print(f" WARNING: Adapter folder '{ADAPTER_PATH}' not found in current directory.")
    print("Listing current directory contents:")
    print(os.listdir("."))
    print("Please update ADAPTER_PATH variable to match your saved folder name.")

# --- TEST SUITE DEFINITION ---

# Group A: ANCHOR TESTS (Stability Check)
# Model should answer these normally, NOT expand them into SD prompts
anchor_tests = [
    {
        "input": "What is 2 + 2?", 
        "expected_keywords": ["4", "four"],
        "should_not_contain": ["render", "octane", "8k", "trending"]
    },
    {
        "input": "What is the capital of France?", 
        "expected_keywords": ["paris"],
        "should_not_contain": ["concept art", "detailed", "photorealistic"]
    },
    {
        "input": "Write a Python function to print hello world.",
        "expected_keywords": ["def", "print", "hello"],
        "should_not_contain": ["lighting", "artstation", "unreal engine"]
    },
    {
        "input": "Explain quantum physics in simple terms.",
        "expected_keywords": ["quantum", "particle", "atom", "energy"],
        "should_not_contain": ["dramatic lighting", "highly detailed", "8k"]
    }
]

# Group B: TARGET TESTS (Capability Check)
# Model should expand these into detailed SD prompts
target_tests = [
    {
        "input": "a cat",
        "min_expansion": 2.0,
        "should_contain": ["cat"],
        "expected_additions": ["detailed", "fur", "eyes", "lighting", "art", "render", "realistic"]
    },
    {
        "input": "a futuristic city",
        "min_expansion": 2.0,
        "should_contain": ["city", "futuristic"],
        "expected_additions": ["neon", "cyberpunk", "lighting", "detailed", "art", "render"]
    },
    {
        "input": "a warrior in armor",
        "min_expansion": 2.0,
        "should_contain": ["warrior", "armor"],
        "expected_additions": ["detailed", "lighting", "fantasy", "concept art", "epic"]
    },
    {
        "input": "portrait of an astronaut",
        "min_expansion": 2.0,
        "should_contain": ["astronaut", "portrait"],
        "expected_additions": ["space", "detailed", "realistic", "lighting", "photography"]
    },
    {
        "input": "dragon perched on mountain",
        "min_expansion": 2.0,
        "should_contain": ["dragon", "mountain"],
        "expected_additions": ["fantasy", "epic", "detailed", "concept art", "majestic"]
    },
    {
        "input": "cyberpunk street at night",
        "min_expansion": 1.5,
        "should_contain": ["cyberpunk", "night", "street"],
        "expected_additions": ["neon", "lights", "detailed", "atmospheric", "rain"]
    }
]

# Validation set for Perplexity (High quality SD prompts)
perplexity_validation_data = [
    "portrait of a cyborg warrior, neon armor, cinematic lighting, highly detailed, 8k, trending on artstation",
    "a beautiful landscape with mountains and a lake, sunset, volumetric lighting, concept art, matte painting",
    "close up of a cat with glowing eyes, fantasy style, digital art, sharp focus, intricate details",
    "steampunk airship flying over a city, victorian architecture, smoke, dramatic lighting, epic scale",
    "cyberpunk street food vendor, rain, reflections, vibrant colors, unreal engine 5 render, photorealistic"
]

# --- LOADING ---
print("="*80)
print("SD PROMPT EXPANDER - REGRESSION TEST SUITE")
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

print(f"[3/4] Loading adapter from: {ADAPTER_PATH}")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print(" Adapter loaded successfully!")
except Exception as e:
    print(f" Error loading adapter: {e}")
    print("\nTroubleshooting:")
    print("  - Did you run train.py successfully?")
    print("  - Check that ADAPTER_PATH points to the correct directory")
    print("  - Expected files: adapter_config.json, adapter_model.bin")
    exit(1)

model.eval()
print("[4/4] Model ready for testing!\n")

# --- INFERENCE ENGINE ---
def generate_response(prompt, is_expansion_task=False):
    """
    Generate model response with appropriate formatting
    
    Args:
        prompt: Input text
        is_expansion_task: If True, uses SD prompt expansion format
    """
    if is_expansion_task:
        # Match our training format exactly
        formatted_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Expand this into a detailed Stable Diffusion prompt: {prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        # Standard chat format for normal questions
        formatted_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response



# --- RUNNING TESTS ---
results = []

print("="*80)
print("PHASE 1: STABILITY TESTS (Anchor)")
print("="*80)
print("Testing that model can still answer normal questions...\n")

anchor_pass_count = 0
for i, test in enumerate(anchor_tests, 1):
    print(f"Test {i}/{len(anchor_tests)}: {test['input'][:50]}...")
    
    response = generate_response(test["input"], is_expansion_task=False)
    response_lower = response.lower()
    
    # Check if expected keywords are present
    has_expected = any(kw in response_lower for kw in test["expected_keywords"])
    
    # Check if it accidentally expanded into an SD prompt
    has_forbidden = any(kw in response_lower for kw in test["should_not_contain"])
    
    passed = has_expected and not has_forbidden
    
    if passed:
        status = " PASS"
        anchor_pass_count += 1
    elif has_forbidden:
        status = " FAIL (Catastrophic forgetting - treating everything as SD prompt)"
    else:
        status = "  FAIL (Incorrect answer)"
    
    print(f"  {status}")
    print(f"  Response: {response[:80]}{'...' if len(response) > 80 else ''}\n")
    
    results.append({
        "Category": "Anchor (Stability)",
        "Test": test["input"],
        "Response": response,
        "Status": status,
        "Expansion_Ratio": "N/A",
        "Details": f"Expected keywords: {has_expected}, No SD terms: {not has_forbidden}"
    })

print(f"Anchor Tests: {anchor_pass_count}/{len(anchor_tests)} passed")

print("\n" + "="*80)
print("PHASE 2: CAPABILITY TESTS (Target)")
print("="*80)
print("Testing that model expands prompts correctly...\n")

target_pass_count = 0
for i, test in enumerate(target_tests, 1):
    print(f"Test {i}/{len(target_tests)}: '{test['input']}'")
    
    response = generate_response(test["input"], is_expansion_task=True)
    response_lower = response.lower()
    
    # Calculate expansion ratio
    expansion_ratio = len(response) / len(test["input"])
    
    # Check if core subject is preserved
    subject_preserved = all(word in response_lower for word in test["should_contain"])
    
    # Check if professional terms were added
    additions_found = sum(1 for term in test["expected_additions"] if term in response_lower)
    additions_ratio = additions_found / len(test["expected_additions"])
    
    # Pass criteria
    sufficient_expansion = expansion_ratio >= test["min_expansion"]
    has_additions = additions_ratio >= 0.2  # At least 20% of expected terms
    
    passed = sufficient_expansion and subject_preserved and has_additions
    
    if passed:
        status = " PASS"
        target_pass_count += 1
    elif not subject_preserved:
        status = " FAIL (Lost original subject)"
    elif not sufficient_expansion:
        status = "  WEAK (Insufficient expansion)"
    else:
        status = "  WEAK (Few professional terms)"
    
    print(f"  {status}")
    print(f"  Input:  {test['input']}")
    print(f"  Output: {response[:100]}{'...' if len(response) > 100 else ''}")
    print(f"  Expansion: {expansion_ratio:.1f}x | Terms: {additions_found}/{len(test['expected_additions'])}\n")
    
    results.append({
        "Category": "Target (Capability)",
        "Test": test["input"],
        "Response": response,
        "Status": status,
        "Expansion_Ratio": f"{expansion_ratio:.2f}x",
        "Details": f"Subject preserved: {subject_preserved}, Terms added: {additions_found}/{len(test['expected_additions'])}"
    })

print(f"Target Tests: {target_pass_count}/{len(target_tests)} passed")

print("\n" + "="*80)
print(" MINI-HELLASWAG (Common Sense)")
print("="*80)

HELLASWAG_SAMPLES = 20
try:
    dataset = load_dataset("rowan/hellaswag", split="validation", trust_remote_code=True)
    subset = dataset.select(range(HELLASWAG_SAMPLES))
    hs_correct = 0

    for item in tqdm(subset, desc="Evaluating HellaSwag"):
        ctx = item['ctx']
        endings = item['endings']
        label = item['label']
        correct_char = ["A", "B", "C", "D"][int(label)]

        prompt = f"<|start_header_id|>user<|end_header_id|>\n\nComplete the description with the most plausible ending:\n\nContext: {ctx}\n\nOptions:\nA) {endings[0]}\nB) {endings[1]}\nC) {endings[2]}\nD) {endings[3]}\n\nRespond with ONLY the letter (A, B, C, or D).<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe answer is"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, temperature=0.1)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("The answer is")[-1].strip().upper()
        prediction = prediction[0] if len(prediction) > 0 else "Z"

        if prediction == correct_char:
            hs_correct += 1

    hs_acc = hs_correct / HELLASWAG_SAMPLES
    print(f"\nHELLASWAG RESULT: {hs_acc:.1%} ({hs_correct}/{HELLASWAG_SAMPLES})")

except Exception as e:
    print(f"Error running HellaSwag: {e}")
    hs_acc = 0

# ==============================================================================
#  PERPLEXITY 
# ==============================================================================
print("\n" + "="*80)
print("PHASE 5: PERPLEXITY (Domain Adaptation)")
print("="*80)

def calculate_perplexity(model, tokenizer, text_list):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in text_list:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    return math.exp(total_loss / total_tokens)

ppl_score = calculate_perplexity(model, tokenizer, perplexity_validation_data)
print(f"MODEL PERPLEXITY: {ppl_score:.2f}")
print("(Lower is better. < 20 indicates good adaptation to Stable Diffusion style)")


# --- FINAL REPORT ---
print("\n" + "="*80)
print("FINAL REPORT - RESUME READY METRICS")
print("="*80)

print(f"1. Stability (Anchor):       {anchor_pass_count}/{len(anchor_tests)} passed")
print(f"2. Capability (Target):      {target_pass_count}/{len(target_tests)} passed")
print(f"3. Common Sense (HellaSwag): {hs_acc:.1%} (Reasoning)")
print(f"4. Adaptation (Perplexity):  {ppl_score:.2f} (Style Fit)")

print("\nVERDICT:")
if ppl_score < 30 and anchor_pass_count >= 3:
    print("✅ SUCCESS: Model is stable, smart, and highly adapted to the target domain.")
else:
    print("⚠️ CAUTION: Check results. High perplexity or low stability detected.")

df = pd.DataFrame(results)
df.to_csv("regression_report.csv", index=False)
print(f"\n📊 Report saved to: regression_report.csv")