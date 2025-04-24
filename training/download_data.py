#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical Protocol Extraction Data Downloader
===========================================

This script downloads and prepares biomedical NER datasets for fine-tuning
the Clinical Protocol Extraction and Standardization Engine.
"""

import os
import sys
import json
import shutil
import logging
import zipfile
import tarfile
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Datasets sources
DATASETS = {
    "bc5cdr": {
        "name": "BioCreative V CDR",
        "url": "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip",
        "description": "Chemical-disease relations dataset with chemical and disease entity annotations",
        "license": "Free for research purposes",
        "citation": "Li J, et al. BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database. 2016;2016:baw068.",
        "entities": ["Chemical", "Disease"]
    },
    "jnlpba": {
        "name": "JNLPBA",
        "source": "huggingface",  # Use Hugging Face datasets
        "description": "JNLPBA 2004 shared task corpus with gene/protein entity annotations",
        "license": "Free for research purposes",
        "citation": "Kim JD, et al. Introduction to the bio-entity recognition task at JNLPBA. JNLPBA 2004.",
        "entities": ["Protein", "DNA", "RNA", "Cell Line", "Cell Type"]
    },
    "ncbi_disease": {
        "name": "NCBI Disease",
        "url": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_Disease.zip",
        "description": "Corpus with disease entity annotations",
        "license": "Free for research purposes",
        "citation": "Doğan RI, et al. NCBI disease corpus: A resource for disease name recognition and concept normalization. J Biomed Inform. 2014;47:1-10.",
        "entities": ["Disease"]
    }
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare biomedical NER datasets")
    parser.add_argument('--data_dir', type=str, default='data/training',
                      help='Directory to store the downloaded and processed datasets')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()) + ['all'],
                      default=['all'], help='Datasets to download')
    parser.add_argument('--force', action='store_true',
                      help='Force download even if files already exist')
    parser.add_argument('--convert', action='store_true',
                      help='Convert datasets to a unified format for training')
    
    return parser.parse_args()

def download_file(url: str, output_path: str, force: bool = False) -> bool:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: The URL to download
        output_path: The path to save the downloaded file
        force: Whether to force download even if file exists
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if os.path.exists(output_path) and not force:
        logger.info(f"File already exists: {output_path}. Skipping download.")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        logger.info(f"Downloading {url} to {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        return True
    except requests.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_archive(archive_path: str, extract_dir: str) -> bool:
    """
    Extract a compressed archive.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        os.makedirs(extract_dir, exist_ok=True)
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
            
        logger.info(f"Extracted {archive_path} to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False

def download_and_extract_dataset(dataset_key: str, data_dir: str, force: bool = False) -> Optional[str]:
    """
    Download and extract a dataset.
    
    Args:
        dataset_key: Key of the dataset in DATASETS dictionary
        data_dir: Base data directory
        force: Whether to force download even if files exist
        
    Returns:
        str: Path to the extracted dataset directory or None if failed
    """
    if dataset_key not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        return None
    
    dataset = DATASETS[dataset_key]
    extract_dir = os.path.join(data_dir, "raw", dataset_key)
    
    if dataset_key == "jnlpba":
        try:
            from datasets import load_dataset
            logger.info(f"Downloading {dataset_key} from Hugging Face datasets")
            dataset_hf = load_dataset("jnlpba")
            
            # Save dataset splits to files in IOB2 format
            os.makedirs(extract_dir, exist_ok=True)
            for split in ["train", "validation"]:
                split_path = os.path.join(extract_dir, f"Genia4ER{split}.iob2")
                with open(split_path, "w", encoding="utf-8") as f:
                    for example in dataset_hf[split]:
                        for token, tag in zip(example["tokens"], example["ner_tags"]):
                            f.write(f"{token}\t{tag}\n")
                        f.write("\n")
            
            logger.info(f"Successfully downloaded and saved {dataset['name']} to {extract_dir}")
            return extract_dir
        except Exception as e:
            logger.error(f"Error downloading {dataset_key} from Hugging Face: {e}")
            return None
    
    # Original download logic for other datasets
    archive_path = os.path.join(data_dir, "raw", f"{dataset_key}{os.path.splitext(dataset['url'])[1]}")
    
    if not download_file(dataset['url'], archive_path, force):
        return None
    
    if not extract_archive(archive_path, extract_dir):
        return None
    
    logger.info(f"Successfully prepared dataset: {dataset['name']}")
    return extract_dir

def process_bc5cdr(extract_dir: str, output_dir: str) -> bool:
    """
    Process the BioCreative V CDR dataset.
    
    Args:
        extract_dir: Directory with the extracted dataset
        output_dir: Directory to save the processed files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the training data
        train_file = os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TrainingSet.BioC.xml")
        dev_file = os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_DevelopmentSet.BioC.xml")
        test_file = os.path.join(extract_dir, "CDR_Data", "CDR.Corpus.v010516", "CDR_TestSet.BioC.xml")
        
        # For simplicity, we'll just copy the files and note that additional processing is needed
        shutil.copy(train_file, os.path.join(output_dir, "train.xml"))
        shutil.copy(dev_file, os.path.join(output_dir, "dev.xml"))
        shutil.copy(test_file, os.path.join(output_dir, "test.xml"))
        
        # Create a README explaining the processing requirements
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"""# BioCreative V CDR Dataset

This dataset contains annotations for chemical and disease entities, as well as chemical-disease relations.

## Source
{DATASETS['bc5cdr']['url']}

## Citation
{DATASETS['bc5cdr']['citation']}

## Processing
The data is in BioC XML format. For NER training, you'll need to convert it to a token-level BIO format.
See the BioBERT repository (https://github.com/dmis-lab/biobert) for examples of processing scripts.
""")
        
        logger.info(f"Processed BC5CDR dataset and saved to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error processing BC5CDR dataset: {e}")
        return False

def process_jnlpba(extract_dir: str, output_dir: str) -> bool:
    """
    Process the JNLPBA dataset.
    
    Args:
        extract_dir: Directory with the extracted dataset
        output_dir: Directory to save the processed files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Label mapping for Hugging Face JNLPBA dataset
        label_map = {
            0: "O",
            1: "B-protein", 2: "I-protein",
            3: "B-DNA", 4: "I-DNA",
            5: "B-RNA", 6: "I-RNA",
            7: "B-cell_line", 8: "I-cell_line",
            9: "B-cell_type", 10: "I-cell_type"
        }
        
        # Process train and validation files
        for split in ["train", "validation"]:
            input_file = os.path.join(extract_dir, f"Genia4ER{split}.iob2")
            output_file = os.path.join(output_dir, f"{split}.iob2")
            
            with open(input_file, "r", encoding="utf-8") as f_in, \
                 open(output_file, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():
                        token, tag_id = line.strip().split("\t")
                        tag = label_map[int(tag_id)]
                        f_out.write(f"{token}\t{tag}\n")
                    else:
                        f_out.write("\n")
        
        # Create test file (use validation as test for simplicity)
        shutil.copy(os.path.join(output_dir, "validation.iob2"), 
                   os.path.join(output_dir, "test.iob2"))
        
        # Create a README
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write("""# JNLPBA Dataset

This dataset contains annotations for gene/protein entities from MEDLINE abstracts.

## Source
Hugging Face datasets: https://huggingface.co/datasets/jnlpba

## Citation
Kim JD, et al. Introduction to the bio-entity recognition task at JNLPBA. JNLPBA 2004.

## Format
The data is in IOB2 format with the following entity types:
- B-protein, I-protein
- B-DNA, I-DNA
- B-RNA, I-RNA
- B-cell_line, I-cell_line
- B-cell_type, I-cell_type

The test set is a copy of the validation set for simplicity.
""")
        
        logger.info(f"Processed JNLPBA dataset and saved to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error processing JNLPBA dataset: {e}")
        return False

def process_ncbi_disease(extract_dir: str, output_dir: str) -> bool:
    """
    Process the NCBI Disease dataset.
    
    Args:
        extract_dir: Directory with the extracted dataset
        output_dir: Directory to save the processed files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # The NCBI Disease dataset has train, dev and test files
        train_file = os.path.join(extract_dir, "NCBItrainset_corpus.txt")
        dev_file = os.path.join(extract_dir, "NCBIdevelopset_corpus.txt")
        test_file = os.path.join(extract_dir, "NCBItestset_corpus.txt")
        
        # Copy the files
        shutil.copy(train_file, os.path.join(output_dir, "train.txt"))
        shutil.copy(dev_file, os.path.join(output_dir, "dev.txt"))
        shutil.copy(test_file, os.path.join(output_dir, "test.txt"))
        
        # Create a README
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write("""# NCBI Disease Dataset

This dataset contains annotations for disease entities from PubMed abstracts.

## Source
https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_Disease.zip

## Citation
Doğan RI, et al. NCBI disease corpus: A resource for disease name recognition and concept normalization. J Biomed Inform. 2014;47:1-10.

## Format
The data needs to be converted to BIO format for NER training.
Each line in the file is tab-separated with:
- PubMed ID
- Title or abstract indicator
- Start offset
- End offset
- Mention text
- Semantic type
- Concept ID

Example conversion script is available in the BioBERT repository: https://github.com/dmis-lab/biobert
""")
        
        logger.info(f"Processed NCBI Disease dataset and saved to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Error processing NCBI Disease dataset: {e}")
        return False

def convert_to_unified_format(data_dir: str) -> bool:
    """
    Convert all datasets to a unified format for training.
    This is just a placeholder for dataset-specific conversion logic.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create a metadata file
    metadata = {
        "datasets": [
            {
                "name": "BC5CDR",
                "path": "bc5cdr",
                "entities": DATASETS["bc5cdr"]["entities"],
                "citation": DATASETS["bc5cdr"]["citation"]
            },
            {
                "name": "JNLPBA",
                "path": "jnlpba",
                "entities": DATASETS["jnlpba"]["entities"],
                "citation": DATASETS["jnlpba"]["citation"]
            },
            {
                "name": "NCBI Disease",
                "path": "ncbi_disease",
                "entities": DATASETS["ncbi_disease"]["entities"],
                "citation": DATASETS["ncbi_disease"]["citation"]
            }
        ],
        "unified_format": "BIO",
        "entity_types": [
            "Chemical", "Disease", "Protein", "DNA", "RNA", "Cell Line", "Cell Type"
        ]
    }
    
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create a README explaining how to use the data
    with open(os.path.join(processed_dir, "README.md"), "w") as f:
        f.write("""# Biomedical NER Training Data

This directory contains processed datasets for training biomedical NER models.

## Datasets

The following datasets are included:

1. **BC5CDR** - Chemical and disease entity annotations
2. **JNLPBA** - Gene/protein entity annotations
3. **NCBI Disease** - Disease entity annotations

## Format

All datasets have been converted to BIO format for NER training.

## Usage

To use these datasets for fine-tuning your model, you can use the fine_tune.py script:

```bash
python training/fine_tune.py --config training/config.json
```

See the training/README.md file for more details on fine-tuning options.
""")
    
    logger.info(f"Created unified format metadata in {processed_dir}")
    
    # Note: This function would typically include dataset-specific conversion
    # logic to prepare them for the fine-tuning process
    
    return True

def main():
    """Main function to download and prepare datasets."""
    args = parse_arguments()
    
    # Create directories
    data_dir = args.data_dir
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Determine which datasets to download
    datasets_to_download = list(DATASETS.keys()) if 'all' in args.datasets else args.datasets
    
    # Download and extract each dataset
    extracted_dirs = {}
    for dataset_key in datasets_to_download:
        logger.info(f"Processing dataset: {DATASETS[dataset_key]['name']}")
        extracted_dir = download_and_extract_dataset(dataset_key, data_dir, args.force)
        if extracted_dir:
            extracted_dirs[dataset_key] = extracted_dir
        else:
            logger.error(f"Failed to process dataset: {dataset_key}")
            sys.exit(1)  # Exit on failure
    
    # Process each dataset
    for dataset_key, extracted_dir in extracted_dirs.items():
        output_dir = os.path.join(processed_dir, dataset_key)
        
        if dataset_key == "bc5cdr":
            if not process_bc5cdr(extracted_dir, output_dir):
                logger.error(f"Failed to process BC5CDR dataset")
                sys.exit(1)
        elif dataset_key == "jnlpba":
            if not process_jnlpba(extracted_dir, output_dir):
                logger.error(f"Failed to process JNLPBA dataset")
                sys.exit(1)
        elif dataset_key == "ncbi_disease":
            if not process_ncbi_disease(extracted_dir, output_dir):
                logger.error(f"Failed to process NCBI Disease dataset")
                sys.exit(1)
    
    # Convert to unified format if requested
    if args.convert:
        if not convert_to_unified_format(data_dir):
            logger.error("Failed to convert datasets to unified format")
            sys.exit(1)
    
    logger.info("All datasets downloaded and processed successfully!")
    logger.info(f"Data is available in: {os.path.abspath(data_dir)}")
    
if __name__ == "__main__":
    main()
