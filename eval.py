import os
import torch
import evaluate
import torch
from datasets import ClassLabel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from dataset_functions import setup_and_process_dataset, format_data

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

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

if __name__ == "__main__":
    login()
    data = setup_and_process_dataset()
    data["validation"] = data.pop("test")
    formatted_data = data.map(format_data)
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)

    # Ground-truth labels
    REFERENCES = formatted_data["validation"]["label"]
    test_data = formatted_data["validation"]
    
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
        print(response_text)
        if do_full_match:
            # Check if response_text is empty or not a valid class label
            if not response_text or response_text.strip() == "":
                return -1
            try:
                return LABEL_FEATURE.str2int(response_text)
            except ValueError:
                # If str2int fails, fall back to partial matching
                for label in BRAIN_CLASSES:
                    if label in response_text or ALT_LABELS[label] in response_text:
                        return LABEL_FEATURE.str2int(label)
                return -1
        for label in BRAIN_CLASSES:
            # Search for `X: tissue type` or `(X) tissue type` in the response
            if label in response_text or ALT_LABELS[label] in response_text:
                return LABEL_FEATURE.str2int(label)
        return -1


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