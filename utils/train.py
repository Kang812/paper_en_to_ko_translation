import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                                       

import re
import torch
from datasets import load_from_disk, Dataset
from model.load_model import create_model
from unsloth import is_bfloat16_supported
import argparse
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only


def main(args):
    # model load
    model, tokenizer = create_model(args.model_name,
                                    args.max_seq_length,
                                    args.r,
                                    args.lora_alpha,)
    
    tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",)
    
    train_dataset = load_from_disk(args.data_path)['train']
    valid_dataset = load_from_disk(args.data_path)['valid']

    def formatting_prompts_func(examples):
        texts = []
        ko_list = examples['ko']
        en_list = examples['en']
        for ko, en in zip(ko_list, en_list):
            user = {'content': en, 'role':'user'}
            assistant = {'content': ko, 'role':'assistant'}

            texts.append(tokenizer.apply_chat_template([user, assistant], 
                                                    tokenize = False,
                                                    add_generation_prompt = False))

        return { "text" : texts, }

    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    
    trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = train_dataset,
            dataset_text_field = "text",
            max_seq_length = args.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = args.per_device_train_batch_size,
                gradient_accumulation_steps = args.gradient_accumulation_steps,
                warmup_steps = args.warmup_steps,
                num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
                save_total_limit = 3,
                #max_steps = 60,
                save_steps = args.save_steps,
                learning_rate = args.learning_rate,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = args.logging_steps,
                optim = args.optim,
                weight_decay = args.weight_decay,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = args.output_dir,
                report_to = "none", # Use this for WandB etc
                dataloader_pin_memory=True, 
            ),
        )
    
    trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",)
    
    trainer_stats = trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model load
    parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-3B-Instruct')
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)

    # dataset
    parser.add_argument('--data_path', type=str, default='/workspace/paper_translation/paper_translation_data/translation_dataset/')
    
    # train parameter
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--optim', type=str, default="adamw_8bit")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default = '/workspace/paper_translation/save_model/')

    # eval parameter
    parser.add_argument('--save_steps', type=int, default=20000)

    # logging step
    parser.add_argument('--logging_steps', type=int, default=100)


    args = parser.parse_args()
    
    print("Argument:")
    for k, v in args.__dict__.items():
        print(f' {k}: {v}')
    
    main(args)
