# Terminology Mapping System

A complete system for mapping clinical terms to standardized medical terminologies like SNOMED CT, LOINC, and RxNorm. This system enables interoperability with healthcare systems by consistently identifying the same medical concepts.

## Features

- **Embedded Databases**: Local SQLite databases for offline terminology mapping
- **Fuzzy Matching**: Advanced fuzzy matching algorithms for handling abbreviations, misspellings, and variants
- **Context-Aware Mapping**: Uses document context to improve mapping accuracy
- **External Service Integration**: Optional fallback to external terminology services (RxNav, UMLS, BioPortal)
- **Custom Mappings**: User-defined mappings that override standard mappings
- **Synonym Expansion**: Comprehensive medical synonym database

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize the terminology databases:
   ```bash
   python setup_terminology.py
   ```

3. Test the terminology mapping:
   ```bash
   python test_terminology_mapping.py
   ```

## Using the Terminology Mapper

```python
from standards.terminology.mapper import TerminologyMapper

# Create a mapper instance
mapper = TerminologyMapper()
mapper.initialize()

# Map terms to different terminology systems
snomed_result = mapper.map_to_snomed("hypertension")
loinc_result = mapper.map_to_loinc("hemoglobin a1c")
rxnorm_result = mapper.map_to_rxnorm("metformin")

# Use context to improve mapping accuracy
context = "Patient has a history of HTN and is taking atorvastatin for hyperlipidemia"
result = mapper.map_term("high blood pressure", "snomed", context)

# Print results
print(f"SNOMED: {snomed_result.get('code')} - {snomed_result.get('display')}")
print(f"LOINC: {loinc_result.get('code')} - {loinc_result.get('display')}")
print(f"RxNorm: {rxnorm_result.get('code')} - {rxnorm_result.get('display')}")
```

## Architecture

The terminology mapping system uses a cascading approach:

1. **Exact Match**: First attempts to find an exact match in the embedded databases
2. **Fuzzy Match**: If no exact match is found, uses fuzzy matching algorithms
3. **External Services**: Falls back to external terminology services if available
4. **Custom Mappings**: User-defined mappings that override other sources

This ensures the system works offline while providing high-quality mappings.

## Configuration

You can configure the terminology mapper with different options:

```python
config = {
    "data_dir": "/path/to/data/terminology",
    "use_fuzzy_matching": True,
    "use_external_services": True,
    "use_rxnav_api": True,
    "umls_api_key": "your-umls-api-key",  # Optional
    "bioportal_api_key": "your-bioportal-api-key",  # Optional
    "ratio_threshold": 90,  # Fuzzy matching threshold
    "token_sort_ratio_threshold": 85  # Token sort ratio threshold
}

mapper = TerminologyMapper(config)
mapper.initialize()
```

## Database Maintenance

To update the terminology databases with the latest terms:

```bash
python -m standards.terminology.db_updater --config=/path/to/config.json
```

This will download and process the latest terminology data from the official sources (requires API credentials for some sources).