# JNLPBA Dataset

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
