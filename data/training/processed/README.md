# Biomedical NER Training Data

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
