# OMOP CDM Integration

This module provides a comprehensive implementation of the OMOP Common Data Model (CDM) integration for the Clinical Protocol Extraction Engine. It converts extracted clinical protocol data into OMOP CDM format, enabling interoperability with research databases and analytics tools that use the OMOP standard.

## Overview

The OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) is a standardized data model for observational healthcare data. This implementation supports OMOP CDM version 5.4 and includes the following tables:

- `CONDITION_OCCURRENCE`: Records of conditions or diagnoses
- `DRUG_EXPOSURE`: Records of drugs or medications
- `PROCEDURE_OCCURRENCE`: Records of procedures
- `OBSERVATION`: Records of observations
- `MEASUREMENT`: Records of measurements and lab tests
- `DEVICE_EXPOSURE`: Records of device usage
- `SPECIMEN`: Records of specimens

## Usage

### Basic Conversion

```python
from standards.omop.converters import OMOPConverter
from standards.terminology.mapper import TerminologyMapper

# Create a terminology mapper
terminology_mapper = TerminologyMapper()

# Configure the OMOP converter
config = {
    "generate_sql": True,
    "sql_dialect": "sqlite"
}

# Initialize the converter
converter = OMOPConverter(
    terminology_mapper=terminology_mapper,
    config=config
)

# Convert protocol data to OMOP format
result = converter.convert(protocol_data)

# Access converted data
omop_tables = result["tables"]
validation_results = result["validation"]
sql_statements = result.get("sql")
```

### Accessing Converted Tables

```python
# Get condition occurrences
condition_records = result["tables"]["condition_occurrence"]

# Get drug exposures
drug_records = result["tables"]["drug_exposure"]

# Get procedure occurrences
procedure_records = result["tables"]["procedure_occurrence"]

# Get observations
observation_records = result["tables"]["observation"]

# Get measurements
measurement_records = result["tables"]["measurement"]

# Get device exposures
device_records = result["tables"]["device_exposure"]

# Get specimens
specimen_records = result["tables"]["specimen"]
```

### SQL Generation

The converter can generate SQL statements for OMOP tables with the `generate_sql` configuration option:

```python
# Generate SQL for SQLite
config = {
    "generate_sql": True, 
    "sql_dialect": "sqlite"
}

# Or for PostgreSQL
config = {
    "generate_sql": True, 
    "sql_dialect": "postgresql"
}

converter = OMOPConverter(config=config)
result = converter.convert(protocol_data)

# Access SQL statements
sql = result["sql"]

# Generate SQL file
with open("omop_tables.sql", "w") as f:
    # Add preamble if exists
    if '_preamble' in sql:
        for setup_stmt in sql['_preamble'].get('setup', []):
            f.write(f"{setup_stmt}\n\n")
    
    # Add table creation statements
    for table_name, statements in sql.items():
        if table_name.startswith('_'):  # Skip special entries
            continue
            
        f.write(f"-- {table_name.upper()} TABLE\n")
        f.write(f"{statements['create']};\n\n")
        
        # Add data insert statements
        if 'inserts' in statements and statements['inserts']:
            f.write(f"-- {table_name} DATA\n")
            for insert in statements['inserts']:
                f.write(f"{insert};\n")
            f.write("\n")
        
        # Add index creation statements if available
        if 'indexes' in statements:
            f.write(f"-- {table_name} INDEXES\n")
            for index in statements['indexes']:
                f.write(f"{index}\n")
            f.write("\n")
```

### Validation Results

The conversion result includes a detailed validation report:

```python
validation = result["validation"]

# Check overall validity
is_valid = validation["valid"]  # True or False

# Check for issues
issues = validation["issues"]   # List of validation issues
warnings = validation["warnings"]  # List of validation warnings

# Get validation summary
summary = validation["summary"]
print(f"Total records: {summary['total_records']}")
print(f"Errors: {summary['error_count']}")
print(f"Warnings: {summary['warning_count']}")

# Get validation by table
table_validation = validation["tables"]
for table, stats in table_validation.items():
    print(f"{table}: {stats['records']} records, {stats['errors']} errors, {stats['warnings']} warnings")
```

## Configuration Options

The `OMOPConverter` accepts the following configuration options:

```python
config = {
    # SQL generation options
    "generate_sql": True,  # Generate SQL statements
    "sql_dialect": "sqlite",  # SQL dialect (sqlite, postgresql, mysql, mssql)
    
    # Terminology mapping options
    "use_fuzzy_matching": True,  # Use fuzzy matching for terminology
    "use_external_services": False,  # Use external terminology services
    
    # Validation options
    "validation_level": "strict",  # Validation level (strict, normal, relaxed)
    
    # Date handling for protocols
    "default_protocol_date": "2023-10-01"  # Default date for protocol entities
}
```

## Key Components

1. **OMOPConverter**: The main converter class that handles the transformation of protocol data to OMOP CDM format
2. **Table Creators**: Methods for creating different OMOP tables from protocol elements
3. **Terminology Mapping**: Integration with the terminology mapping system for standardized codes
4. **Validation**: Schema validation and data integrity checks
5. **SQL Generation**: Utilities for generating SQL statements for different database systems

## Supported Data Types

The converter can handle various types of protocol data:

- **Conditions**: Mapped to CONDITION_OCCURRENCE
- **Medications**: Mapped to DRUG_EXPOSURE
- **Procedures**: Mapped to PROCEDURE_OCCURRENCE
- **Observations**: Mapped to OBSERVATION
- **Measurements**: Mapped to MEASUREMENT
- **Devices**: Mapped to DEVICE_EXPOSURE
- **Specimens**: Mapped to SPECIMEN

## Extensions and Customizations

The converter can be extended in several ways:

1. **Custom Table Mappings**: Add methods to create additional OMOP tables
2. **Additional SQL Dialects**: Extend SQL generation for other database systems
3. **Custom Validation Rules**: Add domain-specific validation rules
4. **Enhanced Terminology Mapping**: Integrate additional vocabulary sources

## OMOP CDM Resources

- [OMOP CDM Documentation](https://ohdsi.github.io/CommonDataModel/)
- [OMOP CDM on GitHub](https://github.com/OHDSI/CommonDataModel)
- [OMOP Vocabulary](https://athena.ohdsi.org/)

## Version Information

This implementation supports OMOP CDM version 5.4. To check the version:

```python
converter = OMOPConverter()
version = converter.get_cdm_version()  # Returns "5.4"
```