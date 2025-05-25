import torch
from unsloth import FastLanguageModel

def create_model(model_name, max_seq_length, r, lora_alpha, random_state = 3407):
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    
    model = FastLanguageModel.get_peft_model(
            model,
            r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = lora_alpha,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    
    return model, tokenizer