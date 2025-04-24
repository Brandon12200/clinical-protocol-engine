# Terminology Databases

This directory contains SQLite databases and configuration files for terminology mapping.

## Database Files

- `snomed_core.sqlite`: Core SNOMED CT concepts for offline mapping
- `loinc_core.sqlite`: Core LOINC concepts for offline mapping
- `rxnorm_core.sqlite`: Core RxNorm concepts for offline mapping
- `custom_mappings.json`: User-defined custom mappings

## Synonyms

The `synonyms` directory contains JSON files with medical synonym sets used for fuzzy matching and term normalization:

- `medical_synonyms.json`: Common medical term synonyms and abbreviations

## Setup and Maintenance

### Initial Setup

To set up the terminology mapping system for the first time, run:

```bash
# Generate and populate the databases with sample data
python setup_terminology.py

# Test the terminology mapping with clinical examples
python test_terminology_mapping.py
```

### Database Updates

These databases are designed to work offline but can be updated periodically using the database update scripts.

```bash
# Update terminology databases (requires API credentials)
python -m standards.terminology.db_updater --config=/path/to/config.json
```

### Configuring External Services

To enable external terminology services like UMLS, BioPortal, or RxNav, create a configuration file:

```json
{
  "use_external_services": true,
  "use_fuzzy_matching": true,
  "use_rxnav_api": true,
  "umls_api_key": "your-umls-api-key",
  "bioportal_api_key": "your-bioportal-api-key"
}
```

Then run:

```bash
python -m standards.terminology.configure_mappings --config=/path/to/config.json
```

## Adding Custom Mappings

Custom mappings can be added directly to the `custom_mappings.json` file or through the application's API.

## Architecture

The terminology mapping system uses a cascading approach:

1. **Exact Match**: First attempts to find an exact match in the embedded databases
2. **Fuzzy Match**: If no exact match is found, uses fuzzy matching algorithms
3. **External Services**: Falls back to external terminology services if available
4. **Custom Mappings**: User-defined mappings that override other sources

This ensures the system works offline while providing high-quality mappings.