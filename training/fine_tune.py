#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical Protocol Extraction Fine-tuning Script
==============================================

This script fine-tunes a pre-trained biomedical language model for 
named entity recognition in clinical protocols, optimized for speed on M1 Pro.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datasets import Dataset, DatasetDict

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset, Dataset, DatasetDict

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for biomedical named entity recognition"
    )
    parser.add_argument(
        "--config", type=str, default="training/config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/training/processed",
        help="Path to processed dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/biobert_model",
        help="Directory to save the fine-tuned model"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def prepare_dataset(data_dir: str, dataset_name: str) -> DatasetDict:
    """
    Prepare the dataset for fine-tuning by loading local IOB2 files.
    
    Args:
        data_dir: Directory containing processed datasets (e.g., 'data/training/processed')
        dataset_name: Name of the dataset (e.g., 'jnlpba')
        
    Returns:
        DatasetDict: Dictionary containing train, validation, and test datasets
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    datasets = {}
    
    # Define splits and corresponding file names
    splits = {
        "train": "train.iob2",
        "validation": "validation.iob2",
        "test": "test.iob2"
    }
    
    for split, filename in splits.items():
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Read IOB2 file
        tokens = []
        labels = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    if current_tokens:
                        tokens.append(current_tokens)
                        labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            
            # Append the last sentence if exists
            if current_tokens:
                tokens.append(current_tokens)
                labels.append(current_labels)
        
        # Create Hugging Face Dataset
        datasets[split] = Dataset.from_dict({
            "tokens": tokens,
            "ner_tags": labels
        })
    
    # Return as DatasetDict
    return DatasetDict(datasets)

def preprocess_data(dataset: Dataset, tokenizer, label_map: Dict[str, int]) -> Dataset:
    """
    Preprocess the dataset for transformers with optimized tokenization.
    
    Args:
        dataset: The input dataset
        tokenizer: Tokenizer for the model
        label_map: Mapping from string labels to integer IDs
        
    Returns:
        Dataset: Processed dataset
    """
    
    def tokenize_and_align_labels(examples):
        """Tokenize inputs and align labels with wordpieces."""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            # Remove max_length and padding; handle dynamically with data collator
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Convert string labels to IDs
    dataset = dataset.map(
        lambda examples: {"ner_tags": [[label_map.get(tag, 0) for tag in example] for example in examples["ner_tags"]]},
        batched=True,
        num_proc=None,  # Disable parallelism to avoid semaphore issues
    )
    
    # Tokenize and align labels
    processed_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=None,  # Disable parallelism to avoid semaphore issues
    )
    
    return processed_dataset

def fine_tune(config: Dict[str, Any], data_dir: str, output_dir: str) -> None:
    """
    Fine-tune a model on biomedical NER data with optimizations for M1 Pro.
    
    Args:
        config: Configuration dictionary
        data_dir: Directory with processed datasets
        output_dir: Directory to save the fine-tuned model
    """
    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Load the pre-trained model and tokenizer
    model_name = config.get("base_model_name", "dmis-lab/biobert-base-cased-v1.2")
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    dataset_name = config.get("dataset", "jnlpba")
    logger.info(f"Preparing dataset: {dataset_name}")
    
    datasets = prepare_dataset(data_dir, dataset_name)
    
    if not datasets:
        logger.error("Failed to prepare dataset. Exiting.")
        return
    
    # Define label mappings for JNLPBA
    labels = [
        "O",
        "B-protein", "I-protein",
        "B-DNA", "I-DNA",
        "B-RNA", "I-RNA",
        "B-cell_line", "I-cell_line",
        "B-cell_type", "I-cell_type"
    ]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    
    # Load model configuration
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Load the model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=model_config,
    )
    
    # Force CPU for stability
    device = torch.device("cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # Verify PyTorch backend
    logger.info(f"MPS available but not using: {torch.backends.mps.is_available()}")
    
    # Preprocess the data
    processed_datasets = DatasetDict({
        split: preprocess_data(datasets[split], tokenizer, label2id)
        for split in datasets.keys()
    })
    
    # Define training arguments with optimizations
    batch_size = config.get("batch_size", 8)
    total_examples = len(processed_datasets["train"])
    steps_per_epoch = total_examples // batch_size
    eval_steps = max(500, steps_per_epoch // 2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=config.get("overwrite_output", True),
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config.get("learning_rate", 5e-5),
        warmup_steps=config.get("warmup_steps", 0),
        weight_decay=config.get("weight_decay", 0.01),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_num_workers=0,  # Disable parallelism to avoid semaphore issues
        use_mps_device=False,  # Force disable MPS
        fp16=False,
        gradient_accumulation_steps=2,  # Accumulate gradients to compensate for small batch size
    )
    
    # Data collator with dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    
    # Metrics computation
    def compute_metrics(eval_preds):
        """Compute NER metrics."""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        from seqeval.metrics import f1_score, precision_score, recall_score
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets.get("train"),
        eval_dataset=processed_datasets.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Log training start time
    import time
    start_time = time.time()
    logger.info("Starting fine-tuning...")
    
    # Train the model with error handling
    try:
        trainer.train()
        
        # Log training duration
        duration = time.time() - start_time
        logger.info(f"Fine-tuning took {duration / 60:.2f} minutes")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    
    # Save the model with explicit state saving
    logger.info(f"Saving model to {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Explicitly save each component
    try:
        # Save model state dict first
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        logger.info(f"Saved model weights to {os.path.join(output_dir, 'pytorch_model.bin')}")
        
        # Save the model config
        model.config.save_pretrained(output_dir)
        logger.info(f"Saved model config to {output_dir}")
        
        # Save the tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved tokenizer to {output_dir}")
        
        # Try the Trainer's save method as well for completeness
        try:
            trainer.save_model(output_dir)
            logger.info("Additional trainer.save_model() successful")
        except Exception as e:
            logger.warning(f"Trainer.save_model() failed but weights were already saved: {e}")
    except Exception as e:
        logger.error(f"Error during model saving: {e}", exc_info=True)
        raise
        
    # Save entity labels
    try:
        with open(os.path.join(output_dir, 'entity_labels.txt'), 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        logger.info(f"Saved entity labels to {os.path.join(output_dir, 'entity_labels.txt')}")
    except Exception as e:
        logger.error(f"Error saving entity labels: {e}", exc_info=True)
    
    logger.info("Fine-tuning complete!")

def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    fine_tune(config, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()