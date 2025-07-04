import os
import torch
import evaluate
from datasets import ClassLabel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login
from dataset_functions import setup_and_process_dataset, format_data

if __name__ == "__main__":
    login()
    data = setup_and_process_dataset()
    data["validation"] = data.pop("test")
    formatted_data = data.map(format_data)
    model_id = "google/medgemma-4b-it"
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    def collate_fn(examples: list[dict[str, any]]):
        texts = []
        images = []
        for example in examples:
            images.append([example["image"].convert("RGB")])
            texts.append(
                processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, with the padding and image tokens masked in
        # the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens that are not used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    num_train_epochs = 3  # @param {type: "number"}
    learning_rate = 2e-4  # @param {type: "number"}

    args = SFTConfig(
        output_dir="medgemma-4b-it-sft-lora-brain-regions",            # Directory and Hub repository id to save the model to
        num_train_epochs=num_train_epochs,                       # Number of training epochs
        per_device_train_batch_size=4,                           # Batch size per device during training
        per_device_eval_batch_size=4,                            # Batch size per device during evaluation
        gradient_accumulation_steps=4,                           # Number of steps before performing a backward/update pass
        gradient_checkpointing=True,                             # Enable gradient checkpointing to reduce memory usage
        optim="adamw_torch_fused",                               # Use fused AdamW optimizer for better performance
        logging_steps=50,                                        # Number of steps between logs
        save_strategy="epoch", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,            # Save checkpoint every epoch
        eval_strategy="epoch",                                   # Evaluate every `eval_steps`
        eval_steps=1,                                           # Number of steps between evaluations
        learning_rate=learning_rate,                             # Learning rate based on QLoRA paper
        bf16=True,                                               # Use bfloat16 precision
        max_grad_norm=0.3,                                       # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                                       # Warmup ratio based on QLoRA paper
        lr_scheduler_type="linear",                              # Use linear learning rate scheduler
        push_to_hub=True,                                        # Push model to Hub
        report_to="tensorboard",                                 # Report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={"skip_prepare_dataset": True},           # Skip default dataset preparation to preprocess manually
        remove_unused_columns = False,                           # Columns are unused for training but needed for data collator
        label_names=["labels"],                                  # Input keys that correspond to the labels
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=formatted_data["train"],
        eval_dataset=formatted_data["validation"].shuffle(), #.select(range(200)),  # Use subset of validation set for faster run
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    trainer.train()


    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    # Ground-truth labels
    REFERENCES = formatted_data["validation"]["label"]
    test_data = formatted_data["validation"]

    def compute_metrics(predictions: list[int]) -> dict[str, float]:
        metrics = {}
        metrics.update(accuracy_metric.compute(
            predictions=predictions,
            references=REFERENCES,
        ))
        metrics.update(f1_metric.compute(
            predictions=predictions,
            references=REFERENCES,
            average="weighted",
        ))
        return metrics
    
    BRAIN_CLASSES = [
        "A: frontal",
        "B: occipital",
        "C: parietal",
        "D: temporal",
    ]
    options = "\n".join(BRAIN_CLASSES)
    PROMPT = f"What brain region does the edema span the most?\n{options}"

    # Rename the class names to the tissue classes, `X: tissue type`
    test_data = test_data.cast_column(
        "label",
        ClassLabel(names=BRAIN_CLASSES)
    )

    LABEL_FEATURE = test_data.features["label"]
    # Mapping to alternative label format, `(X) tissue type`
    ALT_LABELS = dict([
        (label, f"({label.replace(': ', ') ')}") for label in BRAIN_CLASSES
    ])


    def postprocess(prediction: list[dict[str, str]], do_full_match: bool=False) -> int:
        response_text = prediction[0]["generated_text"]
        if do_full_match:
            return LABEL_FEATURE.str2int(response_text)
        for label in BRAIN_CLASSES:
            # Search for `X: tissue type` or `(X) tissue type` in the response
            if label in response_text or ALT_LABELS[label] in response_text:
                return LABEL_FEATURE.str2int(label)
        return -1

    from transformers import pipeline

    pt_pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16,
    )

    # Set `do_sample = False` for deterministic responses
    pt_pipe.model.generation_config.do_sample = False
    pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    pt_outputs = pt_pipe(
        text=test_data["messages"],
        images=test_data["image"],
        max_new_tokens=40,
        batch_size=64,
        return_full_text=False,
    )

    pt_predictions = [postprocess(out) for out in pt_outputs]

    pt_metrics = compute_metrics(pt_predictions)
    print(f"Baseline metrics: {pt_metrics}")

    ft_pipe = pipeline(
        "image-text-to-text",
        model="axel-darmouni/medgemma-4b-it-sft-lora-brain-regions",
        processor=processor,
        torch_dtype=torch.bfloat16,
    )

    # Set `do_sample = False` for deterministic responses
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    # Use left padding during inference
    processor.tokenizer.padding_side = "left"

    ft_outputs = ft_pipe(
        text=test_data["messages"],
        images=test_data["image"],
        max_new_tokens=20,
        batch_size=64,
        return_full_text=False,
    )

    ft_predictions = [postprocess(out, do_full_match=True) for out in ft_outputs]

    ft_metrics = compute_metrics(ft_predictions)
    print(f"Fine-tuned metrics: {ft_metrics}")