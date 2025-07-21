#!/usr/bin/env python3
"""
Systematic debugging of intervention tensor dimension issues
Strategic iteration between implementation and error output
"""

import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pyvene import create_gpt2, IntervenableConfig, RepresentationConfig, IntervenableModel

class InterventionDebugger:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_log = []
        
    def log(self, message, status="INFO"):
        entry = f"[{status}] {message}"
        print(entry)
        self.debug_log.append(entry)
        
    def test_step(self, step_name, test_func):
        """Run a test step and log results"""
        self.log(f"\n=== TESTING: {step_name} ===", "TEST")
        try:
            result = test_func()
            self.log(f"✅ {step_name} PASSED", "PASS")
            return True, result
        except Exception as e:
            self.log(f"❌ {step_name} FAILED: {str(e)}", "FAIL")
            self.log(f"   Error type: {type(e).__name__}", "ERROR")
            return False, str(e)
    
    def debug_model_loading(self):
        """Debug model loading - find the right model type"""
        
        def test_pyvene_model():
            config, tokenizer, model = create_gpt2("gpt2") 
            self.log(f"PyVene model type: {type(model)}")
            
            # Test basic forward pass
            inputs = tokenizer("Hello world", return_tensors="pt")
            outputs = model(**inputs)
            self.log(f"Output type: {type(outputs)}")
            self.log(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            
            has_logits = hasattr(outputs, 'logits')
            self.log(f"Has logits: {has_logits}")
            
            return config, tokenizer, model, has_logits
            
        def test_transformers_model():
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self.log(f"Transformers model type: {type(model)}")
            
            # Test basic forward pass
            inputs = tokenizer("Hello world", return_tensors="pt")
            outputs = model(**inputs)
            self.log(f"Output type: {type(outputs)}")
            
            has_logits = hasattr(outputs, 'logits')
            self.log(f"Has logits: {has_logits}")
            if has_logits:
                self.log(f"Logits shape: {outputs.logits.shape}")
                
            return model, tokenizer, has_logits
        
        # Test both model types
        success1, pyvene_result = self.test_step("PyVene Model Loading", test_pyvene_model)
        success2, transformers_result = self.test_step("Transformers Model Loading", test_transformers_model)
        
        return success1, success2, pyvene_result, transformers_result
    
    def debug_simple_intervention(self):
        """Debug simplest possible intervention"""
        
        def test_basic_intervention():
            # Use transformers model that we know has logits
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = model.to(self.device)
            
            # Create simplest intervention config
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=6,  # Middle layer
                    component="block_output",  # Whole block output
                    unit="pos",  # Position-wise
                    low_rank_dimension=1
                )
            ])
            
            self.log(f"Intervention config created: {config}")
            
            # Create intervenable model
            intervenable = IntervenableModel(config, model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            self.log("Intervenable model created successfully")
            
            # Test with single sentence (no batching issues)
            text = "The cat sat on the"
            base_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            source_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            
            self.log(f"Input shape: {base_inputs['input_ids'].shape}")
            
            # Try simplest intervention spec
            intervention_spec = {"sources->base": ([[[0]]], [[[0]]])}  # Intervene position 0 -> position 0
            
            self.log(f"Intervention spec: {intervention_spec}")
            
            # Run intervention
            _, counterfactual = intervenable(
                base_inputs,
                [source_inputs], 
                intervention_spec
            )
            
            self.log(f"Intervention successful! Output type: {type(counterfactual)}")
            self.log(f"Counterfactual logits shape: {counterfactual.logits.shape}")
            
            return True
            
        return self.test_step("Basic Intervention", test_basic_intervention)
    
    def debug_position_indexing(self):
        """Debug position indexing issues"""
        
        def test_position_variations():
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = model.to(self.device)
            
            config = IntervenableConfig([
                RepresentationConfig(layer=6, component="block_output", unit="pos", low_rank_dimension=1)
            ])
            intervenable = IntervenableModel(config, model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            text = "The cat sat on the mat"
            base_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            source_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            
            seq_len = base_inputs['input_ids'].shape[1]
            self.log(f"Sequence length: {seq_len}")
            
            # Test different position specifications
            position_tests = [
                ("First position", [[[0]]], [[[0]]]),
                ("Last position explicit", [[[seq_len-1]]], [[[seq_len-1]]]),  
                ("Middle position", [[[seq_len//2]]], [[[seq_len//2]]]),
            ]
            
            results = []
            for desc, source_pos, target_pos in position_tests:
                try:
                    spec = {"sources->base": (source_pos, target_pos)}
                    self.log(f"Testing {desc}: {spec}")
                    
                    _, counterfactual = intervenable(base_inputs, [source_inputs], spec)
                    self.log(f"✅ {desc} worked")
                    results.append((desc, True, None))
                except Exception as e:
                    self.log(f"❌ {desc} failed: {str(e)}")
                    results.append((desc, False, str(e)))
            
            # Test the problematic -1 indexing
            try:
                spec = {"sources->base": ([[[-1]]], [[[-1]]])}
                self.log(f"Testing -1 indexing: {spec}")
                _, counterfactual = intervenable(base_inputs, [source_inputs], spec)
                self.log(f"✅ -1 indexing worked!")
                results.append(("-1 indexing", True, None))
            except Exception as e:
                self.log(f"❌ -1 indexing failed: {str(e)}")
                results.append(("-1 indexing", False, str(e)))
                
            return results
            
        return self.test_step("Position Indexing Variations", test_position_variations)
    
    def debug_attention_head_intervention(self):
        """Debug attention head specific interventions"""
        
        def test_attention_head():
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = model.to(self.device)
            
            # Try attention head intervention
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=9,
                    component="attention_output",  # Attention specific
                    unit="head",  # Head-wise
                    max_number_of_units=1,  # Single head
                    low_rank_dimension=1
                )
            ])
            
            self.log(f"Attention head config: {config}")
            
            intervenable = IntervenableModel(config, model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            text = "John gave Mary the book"
            base_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            source_inputs = tokenizer(text, return_tensors="pt").to(self.device)
            
            # Try different head intervention specs
            head_specs = [
                ("Head 0", {"sources->base": ([[[0]]], [[[0]]])}),
                ("Head 1", {"sources->base": ([[[1]]], [[[1]]])}),
            ]
            
            results = []
            for desc, spec in head_specs:
                try:
                    self.log(f"Testing {desc}: {spec}")
                    _, counterfactual = intervenable(base_inputs, [source_inputs], spec)
                    self.log(f"✅ {desc} worked")
                    results.append((desc, True, None))
                except Exception as e:
                    self.log(f"❌ {desc} failed: {str(e)}")
                    results.append((desc, False, str(e)))
                    
            return results
            
        return self.test_step("Attention Head Intervention", test_attention_head)
    
    def debug_batch_processing(self):
        """Debug batch processing issues"""
        
        def test_batch_intervention():
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = model.to(self.device)
            
            config = IntervenableConfig([
                RepresentationConfig(layer=6, component="block_output", unit="pos", low_rank_dimension=1)
            ])
            intervenable = IntervenableModel(config, model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            # Test batch of different lengths
            texts = [
                "The cat sat",
                "John gave Mary the book yesterday",
                "Hello world"
            ]
            
            # Test padding
            batch_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            self.log(f"Batch input shape: {batch_inputs['input_ids'].shape}")
            self.log(f"Attention mask shape: {batch_inputs['attention_mask'].shape}")
            
            # Try batch intervention
            spec = {"sources->base": ([[[0, 0, 0]]], [[[0, 0, 0]]])}  # Batch size 3
            
            _, counterfactual = intervenable(batch_inputs, [batch_inputs], spec)
            self.log(f"✅ Batch intervention worked: {counterfactual.logits.shape}")
            
            return True
            
        return self.test_step("Batch Processing", test_batch_intervention)
    
    def run_full_debug(self):
        """Run complete debugging sequence"""
        self.log("🔧 STARTING SYSTEMATIC INTERVENTION DEBUGGING", "START")
        
        # Step 1: Model loading
        self.log("\n" + "="*60)
        self.log("STEP 1: DEBUG MODEL LOADING")
        success1, success2, pyvene_result, transformers_result = self.debug_model_loading()
        
        if not success2:
            self.log("❌ FATAL: Can't load basic LM model. Stopping.", "FATAL")
            return False
            
        # Step 2: Simple intervention
        self.log("\n" + "="*60) 
        self.log("STEP 2: DEBUG SIMPLE INTERVENTION")
        success, result = self.debug_simple_intervention()
        
        if not success:
            self.log("❌ FATAL: Basic intervention failed. Stopping.", "FATAL")
            return False
            
        # Step 3: Position indexing
        self.log("\n" + "="*60)
        self.log("STEP 3: DEBUG POSITION INDEXING")
        success, results = self.debug_position_indexing()
        
        # Step 4: Attention heads
        self.log("\n" + "="*60)
        self.log("STEP 4: DEBUG ATTENTION HEAD INTERVENTIONS") 
        success, results = self.debug_attention_head_intervention()
        
        # Step 5: Batch processing
        self.log("\n" + "="*60)
        self.log("STEP 5: DEBUG BATCH PROCESSING")
        success, result = self.debug_batch_processing()
        
        # Summary
        self.log("\n" + "="*60)
        self.log("🎯 DEBUGGING COMPLETE", "SUMMARY")
        
        # Save debug log
        with open("debug_log.json", "w") as f:
            json.dump({
                "debug_log": self.debug_log,
                "timestamp": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
                "final_status": "debugging_complete"
            }, f, indent=2)
            
        return True

if __name__ == "__main__":
    debugger = InterventionDebugger()
    debugger.run_full_debug()