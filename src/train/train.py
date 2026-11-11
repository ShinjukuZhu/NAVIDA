import os
os.environ.setdefault("HF_DATASETS_CACHE", "./.cache/huggingface_cache")
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import logging
import transformers
from transformers import (
    set_seed, 
    AutoProcessor, 
    TrainingArguments,
    TrainerCallback,
    Trainer
)
from src.train.checkpointing import latest_complete_checkpoint, FullCheckpointCallback
import datasets
from datasets import load_dataset
import sys
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

processor = None

@dataclass
class DataArguments:
    dataset_name: str = field(metadata={"help": "JSONL path, or comma-separated JSONL paths mixed at their natural sample counts."})

@dataclass
class ModelArguments:
    model_name_or_path: str = None
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    

class LogCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.logger.info(logs)


def uniform_sample_with_ends(data, n):
    # n > 2
    if len(data) <= n:
        return data

    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]

def convert_example(example):
    """
    correct example into "messages" 
    eg:
    {
      "system": "You are a helpful assistant.",
      "conversations": [
          {"from": "user", "value": "How many objects are included in this image?",
           "image_path": "/path/to/image.png"},
          {"from": "assistant", "value": "<think>\nI can see 10 objects\n</think>\n<answer>\n10\n</answer>"}
      ]
    }
    """
    messages = []
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

    for i in range(0,len(example["conversations"]),2):
        if not isinstance(example["conversations"][i]["image"],list):
            example["conversations"][i]["image"] = [example["conversations"][i]["image"]]

        content = []

        if example['task type'] == 'vln':
            content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations'})
            if len(example["conversations"][i]["image"]) > 1:
                content.extend([{"type": "image", "image": item} for item in uniform_sample_with_ends(example["conversations"][i]["image"][:-1],8)])
            else:
                content.append({"type": "image", "image": example["conversations"][i]["image"][-1]} )
            content.append({"type": "text", "text": 'and an image of the current observation'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][-1]})
            item = example["conversations"][i]["value"].split('current observation')
            content.append({"type": "text", "text": item[1]})
        elif example['task type'] == 'trajectory summarization':
            content.append({"type": "text", "text": 'Assume you are a robot designed for navigation. You are provided with captured images sequences'})
            content.extend([{"type": "image", "image": item} for item in uniform_sample_with_ends(example["conversations"][i]["image"],8)])
            item = example["conversations"][i]["value"].split('images sequences')
            content.append({"type": "text", "text": item[1]})
        elif example['task type'] == 'idm':
            content.append({"type": "text", "text": 'Imagine you are a robot programmed for navigation tasks. You have been given an image of current view'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][0]} )
            content.append({"type": "text", "text": 'and an image of the goal view'})
            content.append({"type": "image", "image": example["conversations"][i]["image"][1]} )
            item = example["conversations"][i]["value"].split('goal view. ')
            content.append({"type": "text", "text": item[1]})
        else:
            raise NotImplementedError
        
        messages.append({
                    "role": "user",
                    "content": content
                })

        messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example["conversations"][i+1]["value"]},
                        ]
                })
    
    example["messages"] = messages
    return example



def collate_fn(examples):
    texts = [
        processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=False)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        imgs = [item.resize((308,252)) for item in imgs]
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # Locate assistant spans in the expanded multimodal token sequence.
    # This avoids fixed token counts that change with the model/template.
    labels = torch.full_like(batch["input_ids"], -100)
    start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for row, ids in enumerate(batch["input_ids"].tolist()):
        pos = 0
        while pos <= len(ids) - len(start):
            if ids[pos:pos + len(start)] != start:
                pos += 1
                continue
            first = pos + len(start)
            last = ids.index(end, first)
            labels[row, first:last + 1] = batch["input_ids"][row, first:last + 1]
            pos = last + 1
        labels[row, batch["attention_mask"][row] == 0] = -100
        if not (labels[row] != -100).any():
            raise ValueError("No assistant supervision tokens found")
    batch["labels"] = labels
    return batch




def load_training_dataset(dataset_name):
    sources = [path.strip() for path in dataset_name.split(",")]
    train_parts = [load_dataset("json", data_files=path, split="train") for path in sources]
    return datasets.DatasetDict({"train": datasets.concatenate_datasets(train_parts)})


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, f"train.log"))
        file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S", )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)

    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")


    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = latest_complete_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    dataset = load_training_dataset(data_args.dataset_name)
    dataset = dataset.shuffle(seed=training_args.seed)

    global processor
    processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, use_fast=False
        )


    logger.info("Using AutoProcessor for VLM model.")

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
            torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
            )
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, 
                                                     attn_implementation = model_args.attn_implementation,
                                                     torch_dtype=model_args.torch_dtype)

    attention_backends = {
        key: getattr(config, "_attn_implementation", None)
        for key, config in [("model", model.config),
                            ("text", model.config.text_config),
                            ("vision", model.config.vision_config)]
    }
    print("ATTENTION_BACKENDS", attention_backends, flush=True)
    assert all(value == model_args.attn_implementation for value in attention_backends.values())
    model = model.to(torch_dtype)
    # set accepts_loss_kwargs for loss scaler bug when setting gradient_accumulation_steps > 1
    model.accepts_loss_kwargs = False

    ###################
    #  (Optional) Frozen vision encoder
    ###################
    visual = model.model.visual if hasattr(model.model, "visual") else model.visual
    for p in visual.parameters():
        p.requires_grad = False
    for name, p in visual.named_parameters():
        if "merger" in name:
            p.requires_grad = True
    model.config.use_cache = False

    # model.enable_input_require_grads() # important when using adapter
    logger.info(f"*** Model in {torch_dtype}***")
    

    ############################
    # Initialize the NaVIDA Trainer
    ############################
    # training_args.dataset_kwargs = {
    #     "skip_prepare_dataset": True,
    # }
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=collate_fn,
        processing_class=processor,
        callbacks=[FullCheckpointCallback()],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset": data_args.dataset_name,
        "tags": ["NaVIDA Training"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)