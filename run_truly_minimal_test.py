#!/usr/bin/env python3
"""
Absolutely minimal test - just verify what works
"""

import torch
import json
from pyvene import create_gpt2

def run_truly_minimal_test():
    """Test just the absolute basics"""
    print("🔬 ABSOLUTELY MINIMAL TEST")
    print("=" * 40)
    
    results = {
        "test_type": "truly_minimal",
        "steps_completed": [],
        "failures": [],
        "honest_conclusion": ""
    }
    
    # Step 1: Load model
    try:
        print("Step 1: Loading GPT-2...")
        config, tokenizer, model = create_gpt2("gpt2")
        print("✅ Model loaded successfully")
        results["steps_completed"].append("model_loading")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        results["failures"].append(f"model_loading: {e}")
        return results
    
    # Step 2: Basic tokenization
    try:
        print("Step 2: Testing tokenization...")
        text = "Hello world"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✅ Tokenized '{text}' -> {tokens['input_ids'].shape}")
        results["steps_completed"].append("tokenization")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        results["failures"].append(f"tokenization: {e}")
        return results
    
    # Step 3: Basic model forward pass
    try:
        print("Step 3: Testing model inference...")
        with torch.no_grad():
            outputs = model(**tokens)
            print(f"✅ Model forward pass -> output type: {type(outputs)}")
            print(f"    Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            
            # Try to get logits properly
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"✅ Got logits: {logits.shape}")
                results["steps_completed"].append("inference_with_logits")
            elif hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
                print(f"✅ Got hidden states: {hidden.shape}")
                print("⚠️ No logits attribute - this is GPT2Model not GPT2LMHeadModel")
                results["steps_completed"].append("inference_no_logits")
            else:
                print(f"⚠️ Unclear output format")
                results["steps_completed"].append("inference_unclear")
                
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        results["failures"].append(f"inference: {e}")
        return results
    
    # Step 4: Check if we need LM head
    try:
        print("Step 4: Trying to get proper language model...")
        from transformers import GPT2LMHeadModel
        lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        with torch.no_grad():
            lm_outputs = lm_model(**tokens)
            if hasattr(lm_outputs, 'logits'):
                logits = lm_outputs.logits
                print(f"✅ LM model logits: {logits.shape}")
                
                # Test actual prediction
                last_token_logits = logits[0, -1, :]
                predicted_token_id = torch.argmax(last_token_logits)
                predicted_word = tokenizer.decode(predicted_token_id)
                print(f"✅ Prediction after '{text}': '{predicted_word}'")
                results["steps_completed"].append("lm_prediction")
            else:
                print(f"❌ Still no logits from LM model")
                results["failures"].append("lm_no_logits")
                
    except Exception as e:
        print(f"❌ LM model failed: {e}")
        results["failures"].append(f"lm_model: {e}")
    
    # Honest conclusion
    if "lm_prediction" in results["steps_completed"]:
        results["honest_conclusion"] = "✅ Basic GPT-2 inference works. Can build interventions on this."
    elif "inference_with_logits" in results["steps_completed"]:
        results["honest_conclusion"] = "✅ PyVene model works with logits. Ready for interventions."
    elif "inference_no_logits" in results["steps_completed"]:
        results["honest_conclusion"] = "⚠️ PyVene loads base model, not LM head. Need to fix this for predictions."
    else:
        results["honest_conclusion"] = "❌ Basic inference failed. Major issues to debug."
    
    print(f"\n📊 RESULTS:")
    print(f"Completed: {results['steps_completed']}")
    print(f"Failures: {results['failures']}")
    print(f"Conclusion: {results['honest_conclusion']}")
    
    # Save honest results
    with open("truly_minimal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_truly_minimal_test()